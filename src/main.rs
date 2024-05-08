use bevy::{
    math::Vec3Swizzles,
    prelude::*,
    render::{mesh::*, render_asset::RenderAssetUsages},
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    tasks::ComputeTaskPool,
    window::{PrimaryWindow, WindowResized},
};
use bevy_spatial::{kdtree::KDTree2, AutomaticUpdate, SpatialAccess, SpatialStructure};
use halton::Sequence;
use rand::prelude::*;
use std::time::Duration;

const WINDOW_BOUNDS: Vec2 = Vec2::new(400., 400.);
const BOID_BOUNDS: Vec2 = Vec2::new(WINDOW_BOUNDS.x * 2. / 3., WINDOW_BOUNDS.y * 2. / 3.);
const BOID_COUNT: i32 = 500;
const BOID_SIZE: f32 = 5.;
const BOID_VIS_RANGE: f32 = 40.;
const BOID_VIS_COUNT_MAX: usize = 50;
const BOID_VIS_COUNT_MIN: usize = 10;
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
const BOID_UPDATE_DURATION: f32 = 1. / BOID_UPDATE_FREQ / 2.; // Allocate half of the target framerate towards updating the boids

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: (WINDOW_BOUNDS.x, WINDOW_BOUNDS.y).into(),
                    ..default()
                }),
                ..default()
            }),
            // Track boids in the KD-Tree
            AutomaticUpdate::<SpatialEntity>::new()
                // TODO: check perf of other tree types
                .with_spatial_ds(SpatialStructure::KDTree2)
                .with_frequency(Duration::from_millis((1000. / BOID_UPDATE_FREQ) as u64)),
        ))
        .insert_resource(Time::<Fixed>::from_hz(BOID_UPDATE_FREQ as f64))
        .insert_resource(Bounds(boid_bounds(WINDOW_BOUNDS)))
        .add_event::<DvEvent>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, (flocking_system, velocity_system).chain())
        .add_systems(Update, movement_system)
        .add_systems(Update, (draw_boid_gizmos, bevy::window::close_on_esc))
        .add_systems(Update, update_bounds.run_if(on_event::<WindowResized>()))
        .run();
}

// Marker for entities tracked by KDTree
#[derive(Component, Default)]
struct SpatialEntity;

#[derive(Component)]
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
    kdtree: &Res<KDTree2<SpatialEntity>>,
    boid_query: &Query<(Entity, &Velocity, &Transform), With<SpatialEntity>>,
    boid: &Entity,
    t0: &&Transform,
    neighbor_count: usize,
    cursor: Option<Vec2>,
) -> Vec2 {
    // https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
    let mut dv = Vec2::default();
    let mut vec_away = Vec2::default();
    let mut avg_position = Vec2::default();
    let mut avg_velocity = Vec2::default();
    let mut neighboring_boids = 0;
    let mut close_boids = 0;

    let t0_translation = t0.translation.xy();
    for (_, entity) in kdtree.k_nearest_neighbour(t0_translation, neighbor_count) {
        let Ok((other, v1, t1)) = boid_query.get(entity.unwrap()) else {
            todo!()
        };

        // Don't evaluate against itself
        if *boid == other {
            continue;
        }

        let vec_to = t1.translation.xy() - t0_translation;
        let dist_sq = vec_to.length_squared();
        if dist_sq > BOID_VIS_RANGE * BOID_VIS_RANGE {
            continue;
        }

        // Don't evaluate boids behind
        if let Some(vec_to_norm) = vec_to.try_normalize() {
            let quat_to = Quat::from_rotation_arc_2d(Vec2::X, vec_to_norm);
            let angle = t0.rotation.angle_between(quat_to);
            if angle > BOID_FOV {
                continue;
            }
        }

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
    kdtree: Res<KDTree2<SpatialEntity>>,
    mut neighbor_count: Local<usize>,
    mut dv_event_writer: EventWriter<DvEvent>,
    camera: Query<(&Camera, &GlobalTransform)>,
    window: Query<&Window>,
) {
    let pool = ComputeTaskPool::get();
    let boids = boid_query.iter().collect::<Vec<_>>();
    let boids_per_thread = boids.len().div_ceil(4 * pool.thread_num());

    let mut max_neighbor = *neighbor_count;
    if max_neighbor == 0 {
        // Not yet initialized
        max_neighbor = (BOID_VIS_COUNT_MIN + BOID_VIS_COUNT_MAX) / 2;
    }
    
    let cursor = {
        let (camera, transform) = camera.single();
        window
            .single()
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world_2d(transform, cursor))
    };

    let update_start = std::time::Instant::now();
    // https://docs.rs/bevy/latest/bevy/tasks/struct.ComputeTaskPool.html
    for batch in pool.scope(|s| {
        for chunk in boids.chunks(boids_per_thread) {
            let kdtree = &kdtree;
            let boid_query = &boid_query;
            let camera = &camera;
            let window = &window;

            s.spawn(async move {
                let mut dv_batch: Vec<DvEvent> = vec![];

                for (boid, _, t0) in chunk {
                    dv_batch.push(DvEvent(
                        *boid,
                        flocking_dv(kdtree, boid_query, camera, window, boid, t0, max_neighbor),
                    ));
                }

                dv_batch
            });
        }
    }) {
        dv_event_writer.send_batch(batch);
    }

    let update_end = std::time::Instant::now();
    if (max_neighbor > BOID_VIS_COUNT_MIN)
        && (update_end - update_start).as_secs_f32() > BOID_UPDATE_DURATION * 1.1
    {
        *neighbor_count = (max_neighbor * 9 / 10).max(BOID_VIS_COUNT_MIN);
        println!(
            "Reducing neighbors from {max_neighbor} to {}",
            *neighbor_count
        );
    } else if (max_neighbor < BOID_VIS_COUNT_MAX)
        && ((update_end - update_start).as_secs_f32() < BOID_UPDATE_DURATION * 0.9)
    {
        *neighbor_count = (max_neighbor * 11 / 10).min(BOID_VIS_COUNT_MAX);
        println!(
            "Increasing neighbors from {max_neighbor} to {}",
            *neighbor_count
        );
    }
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
