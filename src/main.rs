use bevy::{
    core::TaskPoolThreadAssignmentPolicy,
    math::Vec3Swizzles,
    prelude::*,
    render::{mesh::*, render_asset::RenderAssetUsages},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    tasks::{block_on, poll_once, AsyncComputeTaskPool, Task},
    time::Stopwatch,
    utils::hashbrown::HashMap,
    window::{PrimaryWindow, WindowResized},
};
use halton::Sequence;
use kiddo::{float::kdtree::KdTree, NearestNeighbour, SquaredEuclidean};
use rand::prelude::*;
use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

const WINDOW_BOUNDS: Vec2 = Vec2::new(400., 400.);
const BOID_BOUNDS: Vec2 = Vec2::new(WINDOW_BOUNDS.x * 2. / 3., WINDOW_BOUNDS.y * 2. / 3.);
const BOID_COUNT: i32 = 500;
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

#[derive(Component, Clone, Copy)]
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

type BoidSpatialTree = KdTree<f32, u64, 2, 32, u32>;

#[derive(Resource)]
struct SpatialKiddo {
    tree: Arc<RwLock<BoidSpatialTree>>,
    task: Option<Task<Vec<DvEvent>>>,
    stopwatch: Stopwatch,
}
impl Default for SpatialKiddo {
    fn default() -> Self {
        Self {
            tree: Arc::new(RwLock::new(BoidSpatialTree::with_capacity(BOID_COUNT))),
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

    for ((x, y), z) in seq {
        let spawn = Vec2::new(x as f32, y as f32) * bounds.0 - bounds.0 / 2.0;

        let mut transform = Transform::from_translation(spawn.extend(z as f32 / BOID_COUNT as f32))
            .with_scale(Vec3::splat(BOID_SIZE));

        transform.rotate_z(0.0);

        let velocity = Velocity(Vec2::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ));

        commands.spawn((
            BoidBundle {
                mesh: MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(
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
                    ),
                    material: materials.add(
                        // Random color for each boid
                        Color::hsl(360. * rng.gen::<f32>(), rng.gen(), 0.7),
                    ),
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
    kdtree: &Arc<RwLock<BoidSpatialTree>>,
    boids: &Arc<HashMap<Entity, (Velocity, Transform)>>,
    boid: Entity,
    cursor: Option<Vec2>,
) -> Vec2 {
    // https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
    let mut dv = Vec2::default();
    let mut vec_away = Vec2::default();
    let mut avg_position = Vec2::default();
    let mut avg_velocity = Vec2::default();
    let mut neighboring_boids = 0;
    let mut close_boids = 0;

    let kdtree = kdtree.read().unwrap();

    let Some(t0) = boids.get(&boid).map(|boid| boid.1) else {
        return Vec2::ZERO;
    };

    let t0_translation = t0.translation.xy();
    let neighbors = kdtree.nearest_n_within::<SquaredEuclidean>(
        &t0_translation.to_array(),
        BOID_VIS_RANGE * BOID_VIS_RANGE,
        BOID_VIS_COUNT,
        true, // In Kiddo 4.2.0, when unsorted, it behaves the same as `within_unsorted`
    );
    for NearestNeighbour { item, .. } in neighbors.into_iter() {
        let other = Entity::from_bits(item);

        // Don't evaluate against itself
        if boid == other {
            continue;
        }

        let Some((v1, t1)) = boids.get(&other) else {
            todo!()
        };

        let vec_to = t1.translation.xy() - t0_translation;

        // // Don't evaluate boids behind
        let vec_to_norm_opt = vec_to.try_normalize();
        if let Some(vec_to_norm) = vec_to_norm_opt {
            let quat_to = Quat::from_rotation_arc_2d(Vec2::X, vec_to_norm);
            let angle = t0.rotation.angle_between(quat_to);
            if angle > BOID_FOV {
                continue;
            }
        }

        let dist_sq = vec_to.length_squared();
        if dist_sq < BOID_PROT_RANGE_SQ {
            // separation
            vec_away -= vec_to;
            close_boids += 1;
        } else {
            // cohesion
            avg_position += vec_to;
            // alignment
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
        let to_cursor = cursor - t0_translation;
        dv += to_cursor * BOID_MOUSE_CHASE_FACTOR;
    }

    dv
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
            .map(|query_data| (query_data.0, (*query_data.1, *query_data.2)))
            .collect::<HashMap<_, _>>(),
    );
    let entities = boids.keys().copied().collect::<Vec<_>>();
    let boids_per_thread = boids.len().div_ceil(pool.thread_num());

    let cursor = {
        let (camera, transform) = camera.single();
        window
            .single()
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world_2d(transform, cursor))
    };

    let tasks = entities
        .chunks(boids_per_thread)
        .enumerate()
        .map(|(_id, chunk)| {
            let boids = Arc::clone(&boids);
            let kdtree = Arc::clone(&spatial.tree);

            let chunk = Vec::from(chunk);

            pool.spawn(async move {
                let mut dv_batch: Vec<DvEvent> = vec![];

                for boid in chunk.into_iter() {
                    dv_batch.push(DvEvent(boid, flocking_dv(&kdtree, &boids, boid, cursor)));
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

    // It's faster to recreate the tree than to update (I measured ~0.4ms vs ~1.6ms)
    *kdtree = Default::default();
    for (entity, transform) in entities.iter() {
        kdtree.add(&transform.translation.xy().to_array(), entity.to_bits());
    }

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
