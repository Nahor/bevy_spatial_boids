use bevy::{
    core::TaskPoolThreadAssignmentPolicy,
    prelude::*,
    render::{mesh::*, render_asset::RenderAssetUsages},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    tasks::{block_on, poll_once, AsyncComputeTaskPool, Task},
    time::Stopwatch,
    utils::hashbrown::HashMap,
    window::{PrimaryWindow, WindowResized},
};
use bevy_spatial_boids::range_chunks::RangeChunk;
use halton::Sequence;
use kd_tree::{ItemAndDistance, KdMap};
use rand::prelude::*;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

const WINDOW_BOUNDS: Vec2 = Vec2::new(400., 400.);
const BOID_COUNT: usize = 500;
const BOID_SIZE: f32 = 5.;
const BOID_VIS_RANGE: f32 = 40.;
const BOID_VIS_COUNT: usize = 100;
const BOID_PROT_RANGE: f32 = 8.;
// https://en.wikipedia.org/wiki/Bird_vision#Extraocular_anatomy
const BOID_FOV: f32 = 120. * std::f32::consts::PI / 180.;
const BOID_PROT_RANGE_SQ: f32 = BOID_PROT_RANGE * BOID_PROT_RANGE;
const BOID_CENTER_FACTOR: f32 = 0.0005;
const BOID_MATCHING_FACTOR: f32 = 0.05;
const BOID_AVOID_FACTOR: f32 = 0.05;
const BOID_TURN_FACTOR: f32 = 0.2;
const BOID_MOUSE_CHASE_FACTOR: f32 = 0.0005;
const BOID_MIN_SPEED: f32 = 2.0;
const BOID_MAX_SPEED: f32 = 4.0;
const BOID_UPDATE_FREQ: f32 = 60.0;
const MATERIAL_COUNT: usize = 12; // 3 primary + 3 secondary + 6 in-between

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: (WINDOW_BOUNDS.x, WINDOW_BOUNDS.y).into(),
                    ..default()
                }),
                ..default()
            })
            .set(TaskPoolPlugin {
                task_pool_options: TaskPoolOptions {
                    async_compute: TaskPoolThreadAssignmentPolicy {
                        min_threads: 4,
                        max_threads: std::usize::MAX,
                        percent: 0.75,
                    },
                    // keep the defaults for everything else
                    ..default()
                },
            }),))
        .insert_resource(Bounds(boid_bounds(WINDOW_BOUNDS)))
        .init_resource::<SpatialKiddo>()
        .add_event::<DvEvent>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                movement_system,
                // order:
                // update_spatial - to handle previous update and free the async tasks
                // velocity_system - then update the positions and velocities
                // flocking_system - then start the next update if necessary
                (update_spatial, velocity_system, flocking_system).chain(),
            ),
        )
        .add_systems(Update, (draw_boid_gizmos, bevy::window::close_on_esc))
        .add_systems(Update, update_bounds.run_if(on_event::<WindowResized>()))
        .run();
}

// Marker for entities tracked by KDTree
#[derive(Component, Default)]
struct SpatialEntity;

#[derive(Component, Clone, Copy, Default)]
struct Velocity(Vec2);

#[derive(Bundle)]
struct BoidBundle {
    mesh: MaterialMesh2dBundle<ColorMaterial>,
    velocity: Velocity,
}

impl Default for BoidBundle {
    fn default() -> Self {
        Self {
            mesh: Default::default(),
            velocity: Velocity(Vec2::default()),
        }
    }
}

// Event for a change of velocity on some boid
#[derive(Event)]
struct DvEvent(Entity, Vec2);

#[derive(Resource, Debug)]
struct Bounds(Vec2);

type BoidSpatialTree = KdMap<[f32; 2], u64>;

#[derive(Resource)]
struct SpatialKiddo {
    tree: Arc<RwLock<BoidSpatialTree>>,
    task: Option<Task<Vec<DvEvent>>>,
    stopwatch: Stopwatch,
}
impl Default for SpatialKiddo {
    fn default() -> Self {
        Self {
            tree: Arc::new(RwLock::new(BoidSpatialTree::default())),
            task: None,
            stopwatch: Stopwatch::new(),
        }
    }
}

fn boid_bounds(window_size: Vec2) -> Vec2 {
    let f = |x: f32| x / (x + 1.0).ln();
    Vec2::new(
        window_size.x - f(window_size.x),
        window_size.y - f(window_size.y),
    )
}

fn update_bounds(windows: Query<&Window, With<PrimaryWindow>>, mut bounds: ResMut<Bounds>) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    bounds.0 = boid_bounds(Vec2::new(window.width(), window.height()));
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    bounds: Res<Bounds>,
) {
    commands.spawn(Camera2dBundle::default());

    let mut rng = rand::thread_rng();

    // Halton sequence for Boid spawns
    let seq = halton::Sequence::new(2)
        .zip(Sequence::new(3))
        .zip(0..BOID_COUNT);

    let materials_list = (0..MATERIAL_COUNT)
        .map(|i| {
            materials.add(Color::hsl(
                360. * i as f32 / MATERIAL_COUNT as f32,
                1.0,
                0.7,
            ))
        })
        .collect::<Vec<_>>();
    let mesh = Mesh2dHandle(
        meshes.add(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                vec![
                    [-0.5, 0.5, 0.0],
                    [1.0, 0.0, 0.0],
                    [-0.5, -0.5, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            )
            .with_inserted_indices(Indices::U32(vec![1, 3, 0, 1, 2, 3])),
        ),
    );

    for ((x, y), seq) in seq {
        let spawn = Vec2::new(x as f32, y as f32) * bounds.0 - bounds.0 / 2.0;

        // For efficient auto-batching, we must have a limited number of
        // `(mesh, material)` combinations, and group the boids with the same
        // combination in sequence
        let mesh = mesh.clone();
        let material_idx = seq * MATERIAL_COUNT / BOID_COUNT;
        let material = materials_list[material_idx].clone();

        // To avoid z-fighting, set a different z-height for each boid.
        // But we also need to avoid all the boids with the same material to
        // be at the same z-height.
        // So spread the height such that we get in spawn order (and using
        // 10 materials to simplify the example)
        //   [ 0, 10, 20, ..., X0,  1, 11, 21, ..., X1,  2, 12, 22, ..., X2,  3, ...]
        // with materials:
        //   [m0, m0, m0, ..., m0, m1, m1, m1, ..., m1, m2, m2, m2, ..., m2, m3, ...]
        // such that, once ordered by z-height, we get:
        //   [ 0,  1,  2,  3, .., 10, 11, 12, ..., 20, 21, 22, ..., X0, X1, X2, ... ]
        //   [m0, m1, m2, m3,..., m0, m1, m2, ..., m0, m1, m2, ..., m0, m1, m2, ... ]
        const MAJOR_TICK: usize = BOID_COUNT / MATERIAL_COUNT; // the X in the comment above
        let height_seq = (seq % MATERIAL_COUNT) * MAJOR_TICK + (seq / MATERIAL_COUNT);
        let transform =
            Transform::from_translation(spawn.extend(height_seq as f32 / BOID_COUNT as f32))
                .with_scale(Vec3::splat(BOID_SIZE));

        let velocity = Velocity(Vec2::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ));

        commands.spawn((
            BoidBundle {
                mesh: MaterialMesh2dBundle {
                    mesh,
                    material,
                    transform,
                    ..default()
                },
                velocity,
            },
            SpatialEntity,
        ));
    }
}

fn draw_boid_gizmos(mut gizmos: Gizmos, bounds: Res<Bounds>) {
    gizmos.rect_2d(Vec2::ZERO, 0.0, bounds.0, Color::GRAY);
}

fn flocking_dv(
    idx: usize,
    kdtree: &Arc<RwLock<BoidSpatialTree>>,
    boids: &Arc<HashMap<Entity, Velocity>>,
    cursor: Option<Vec2>,
) -> Option<(Entity, Vec2)> {
    // https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
    let mut dv = Vec2::default();
    let mut vec_away = Vec2::default();
    let mut avg_position = Vec2::default();
    let mut avg_velocity = Vec2::default();
    let mut neighboring_boids = 0;
    let mut close_boids = 0;

    let kdtree = kdtree.read().unwrap();
    let &(t0_array, boid_bits) = kdtree.get(idx)?;
    let boid = Entity::from_bits(boid_bits);
    let t0 = Vec2::from(t0_array);
    let v0 = boids.get(&boid).copied().unwrap_or_default();

    let neighbors = kdtree.nearests(&t0_array, BOID_VIS_COUNT);
    for ItemAndDistance {
        item: &(t1, item),
        squared_distance: dist_sq,
    } in neighbors.into_iter()
    {
        let other = Entity::from_bits(item);

        // Don't evaluate against itself
        if boid == other {
            continue;
        }

        if dist_sq > BOID_VIS_RANGE * BOID_VIS_RANGE {
            continue;
        }

        let vec_to = Vec2::from(t1) - t0;

        // Don't evaluate boids behind
        let angle = (v0.0.dot(vec_to) * v0.0.length_recip() * vec_to.length_recip()).acos();
        if angle > BOID_FOV {
            continue;
        }

        if dist_sq < BOID_PROT_RANGE_SQ {
            // separation
            vec_away -= vec_to;
            close_boids += 1;
        } else {
            // cohesion
            avg_position += vec_to;
            // alignment
            let v1 = boids.get(&other).copied().unwrap_or_default();
            avg_velocity += v1.0;
            neighboring_boids += 1;
        }
    }

    if neighboring_boids > 0 {
        let neighbors = neighboring_boids as f32;
        dv += avg_position / neighbors * BOID_CENTER_FACTOR;
        dv += avg_velocity / neighbors * BOID_MATCHING_FACTOR;
    }

    if close_boids > 0 {
        let close = close_boids as f32;
        dv += vec_away / close * BOID_AVOID_FACTOR;
    }

    // Chase the mouse
    if let Some(cursor) = cursor {
        let to_cursor = cursor - t0;
        dv += to_cursor * BOID_MOUSE_CHASE_FACTOR;
    }

    Some((boid, dv))
}

fn flocking_system(
    boid_query: Query<(Entity, &Velocity, &Transform), With<SpatialEntity>>,
    mut spatial: ResMut<SpatialKiddo>,
    camera: Query<(&Camera, &GlobalTransform)>,
    window: Query<&Window>,
    time: Res<Time>,
    mut is_late: Local<bool>,
) {
    let expected_elapsed = Duration::from_secs_f32(1.0 / BOID_UPDATE_FREQ);
    spatial.stopwatch.tick(time.delta());
    if spatial.stopwatch.elapsed() < expected_elapsed {
        // Timer hasn't expired yet, we can't start the next update
        return;
    }
    if spatial.task.is_some() {
        // Timer has expired, but we are still processing the previous update
        *is_late = true;
        return;
    }

    if *is_late {
        info!(
            "Late by {:.3?}!!",
            spatial.stopwatch.elapsed() - expected_elapsed
        );
        *is_late = false;
    }

    let mut new_elapsed = (spatial.stopwatch.elapsed() - expected_elapsed).max(Duration::ZERO);
    if new_elapsed > expected_elapsed / 2 {
        // We are already more than half expired, restart from zero
        new_elapsed = Duration::ZERO
    }
    spatial.stopwatch.set_elapsed(new_elapsed);

    let pool = AsyncComputeTaskPool::get();
    let boids = Arc::new(
        boid_query
            .iter()
            .map(|query_data| (query_data.0, *query_data.1))
            .collect::<HashMap<_, _>>(),
    );
    let boids_per_thread = boids.len().div_ceil(pool.thread_num());

    let cursor = {
        let (camera, transform) = camera.single();
        window
            .single()
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world_2d(transform, cursor))
    };

    let tasks = (0..boids.len())
        .chunks(boids_per_thread)
        .enumerate()
        .map(|(_id, chunk)| {
            let boids = Arc::clone(&boids);
            let kdtree = Arc::clone(&spatial.tree);

            pool.spawn(async move {
                let mut dv_batch: Vec<DvEvent> = vec![];

                for idx in chunk {
                    let Some((boid, dv)) = flocking_dv(idx, &kdtree, &boids, cursor) else {
                        continue;
                    };
                    dv_batch.push(DvEvent(boid, dv));
                }

                dv_batch
            })
        })
        .collect::<Vec<_>>();

    spatial.task = Some(pool.spawn(async move {
        let mut events = Vec::new();
        for task in tasks {
            let v = task.await;
            events.extend(v);
        }
        events
    }));
}

fn velocity_system(
    mut events: EventReader<DvEvent>,
    mut boids: Query<(&mut Velocity, &mut Transform)>,
    bounds: Res<Bounds>,
) {
    let half_bound = bounds.0 / 2.;

    for DvEvent(boid, dv) in events.read() {
        let Ok((mut velocity, transform)) = boids.get_mut(*boid) else {
            todo!()
        };

        velocity.0 += *dv;

        let top_left_mask = transform.translation.xy().cmplt(-half_bound);
        let bottom_right_mask = transform.translation.xy().cmpgt(half_bound);

        let turn_factor = Vec2::select(top_left_mask, Vec2::splat(BOID_TURN_FACTOR), Vec2::ZERO)
            - Vec2::select(bottom_right_mask, Vec2::splat(BOID_TURN_FACTOR), Vec2::ZERO);
        velocity.0 = (velocity.0 + turn_factor).clamp_length(BOID_MIN_SPEED, BOID_MAX_SPEED);
    }
}

fn movement_system(mut query: Query<(&Velocity, &mut Transform)>, time: Res<Time>) {
    for (velocity, mut transform) in query.iter_mut() {
        if let Some(velocity_norm) = velocity.0.try_normalize() {
            transform.rotation = Quat::from_rotation_arc_2d(Vec2::X, velocity_norm);
        }
        // The velocity is computed for the fixed update, so we need to scale
        // the variable update to match, capped to avoid moving more than what
        // the algorithm expected (low FPS)
        // In other words, smooth the movement at high FPS without messing it
        // at low FPS.
        let time_delta = (time.delta_seconds() * BOID_UPDATE_FREQ).min(1.0);
        transform.translation.x += velocity.0.x * time_delta;
        transform.translation.y += velocity.0.y * time_delta;
    }
}

fn update_spatial(
    mut tree: ResMut<SpatialKiddo>,
    entities: Query<(Entity, &Transform), With<SpatialEntity>>,
    mut dv_event_writer: EventWriter<DvEvent>,
    mut update: Local<(Timer, Duration, u32)>,
    time: Res<Time>,
) {
    update.0.tick(time.delta());

    let tree = tree.as_mut();

    let Some(events) = tree
        .task
        .as_mut()
        .and_then(|task| block_on(poll_once(task)))
    else {
        return;
    };
    tree.task = None;
    dv_event_writer.send_batch(events);

    // info!("Process");

    let s = std::time::Instant::now();

    let mut kdtree = tree.tree.write().unwrap();

    *kdtree = BoidSpatialTree::build_by_ordered_float(
        entities
            .iter()
            .map(|(entity, transform)| (transform.translation.xy().to_array(), entity.to_bits()))
            .collect::<Vec<_>>(),
    );

    let e = std::time::Instant::now();
    update.1 += e - s;
    update.2 += 1;
    if update.0.finished() {
        info!(
            "Update speed: {:?} freq {:.1}",
            (update.1 / update.2),
            update.2 as f32 / time.elapsed_seconds()
        );
        update.0 = Timer::new(Duration::from_secs(1), TimerMode::Once);
    }
}
