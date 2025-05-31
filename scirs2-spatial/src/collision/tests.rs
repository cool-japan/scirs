//! Tests for the collision module

#[cfg(test)]
mod collision_tests {
    use super::super::continuous::*;
    use super::super::narrowphase::*;
    use super::super::shapes::*;

    #[test]
    fn test_point_circle_collision() {
        let circle = Circle {
            center: [0.0, 0.0],
            radius: 2.0,
        };

        // Test points inside, on boundary, and outside
        assert!(point_circle_collision(&[0.0, 0.0], &circle)); // Center
        assert!(point_circle_collision(&[1.0, 1.0], &circle)); // Inside
        assert!(point_circle_collision(&[2.0, 0.0], &circle)); // On boundary
        assert!(!point_circle_collision(&[3.0, 0.0], &circle)); // Outside
    }

    #[test]
    fn test_point_box2d_collision() {
        let box2d = Box2D {
            min: [-1.0, -1.0],
            max: [1.0, 1.0],
        };

        // Test points inside, on boundary, and outside
        assert!(point_box2d_collision(&[0.0, 0.0], &box2d)); // Inside
        assert!(point_box2d_collision(&[1.0, 0.0], &box2d)); // On boundary
        assert!(!point_box2d_collision(&[2.0, 0.0], &box2d)); // Outside
    }

    #[test]
    fn test_point_sphere_collision() {
        let sphere = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };

        // Test points inside, on boundary, and outside
        assert!(point_sphere_collision(&[0.0, 0.0, 0.0], &sphere)); // Center
        assert!(point_sphere_collision(&[1.0, 1.0, 1.0], &sphere)); // Inside
        assert!(point_sphere_collision(&[0.0, 0.0, 2.0], &sphere)); // On boundary
        assert!(!point_sphere_collision(&[0.0, 0.0, 3.0], &sphere)); // Outside
    }

    #[test]
    fn test_point_box3d_collision() {
        let box3d = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        // Test points inside, on boundary, and outside
        assert!(point_box3d_collision(&[0.0, 0.0, 0.0], &box3d)); // Inside
        assert!(point_box3d_collision(&[1.0, 0.0, 0.0], &box3d)); // On boundary
        assert!(!point_box3d_collision(&[2.0, 0.0, 0.0], &box3d)); // Outside
    }

    #[test]
    fn test_circle_circle_collision() {
        let circle1 = Circle {
            center: [0.0, 0.0],
            radius: 2.0,
        };

        let circle2 = Circle {
            center: [3.0, 0.0],
            radius: 1.5,
        };

        let circle3 = Circle {
            center: [5.0, 0.0],
            radius: 1.0,
        };

        // Test various combinations
        assert!(circle_circle_collision(&circle1, &circle1)); // Same circle
        assert!(circle_circle_collision(&circle1, &circle2)); // Overlapping
        assert!(!circle_circle_collision(&circle1, &circle3)); // Not overlapping
    }

    #[test]
    fn test_sphere_sphere_collision() {
        let sphere1 = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };

        let sphere2 = Sphere {
            center: [3.0, 0.0, 0.0],
            radius: 1.5,
        };

        let sphere3 = Sphere {
            center: [5.0, 0.0, 0.0],
            radius: 1.0,
        };

        // Test various combinations
        assert!(sphere_sphere_collision(&sphere1, &sphere1)); // Same sphere
        assert!(sphere_sphere_collision(&sphere1, &sphere2)); // Overlapping
        assert!(!sphere_sphere_collision(&sphere1, &sphere3)); // Not overlapping
    }

    #[test]
    fn test_box2d_box2d_collision() {
        let box1 = Box2D {
            min: [-1.0, -1.0],
            max: [1.0, 1.0],
        };

        let box2 = Box2D {
            min: [0.5, 0.5],
            max: [2.5, 2.5],
        };

        let box3 = Box2D {
            min: [2.0, 2.0],
            max: [3.0, 3.0],
        };

        // Test various combinations
        assert!(box2d_box2d_collision(&box1, &box1)); // Same box
        assert!(box2d_box2d_collision(&box1, &box2)); // Overlapping
        assert!(!box2d_box2d_collision(&box1, &box3)); // Not overlapping
    }

    #[test]
    fn test_box3d_box3d_collision() {
        let box1 = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        let box2 = Box3D {
            min: [0.5, 0.5, 0.5],
            max: [2.5, 2.5, 2.5],
        };

        let box3 = Box3D {
            min: [2.0, 2.0, 2.0],
            max: [3.0, 3.0, 3.0],
        };

        // Test various combinations
        assert!(box3d_box3d_collision(&box1, &box1)); // Same box
        assert!(box3d_box3d_collision(&box1, &box2)); // Overlapping
        assert!(!box3d_box3d_collision(&box1, &box3)); // Not overlapping
    }

    #[test]
    fn test_ray_sphere_collision() {
        let sphere = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };

        // Ray pointing directly at the sphere
        let ray_origin1 = [5.0, 0.0, 0.0];
        let ray_direction1 = [-1.0, 0.0, 0.0];

        // Ray that misses the sphere
        let ray_origin2 = [5.0, 2.0, 0.0];
        let ray_direction2 = [-1.0, 0.0, 0.0];

        // Ray that starts inside the sphere
        let ray_origin3 = [0.5, 0.0, 0.0];
        let ray_direction3 = [1.0, 0.0, 0.0];

        assert!(ray_sphere_collision(&ray_origin1, &ray_direction1, &sphere).is_some());
        assert!(ray_sphere_collision(&ray_origin2, &ray_direction2, &sphere).is_none());
        assert!(ray_sphere_collision(&ray_origin3, &ray_direction3, &sphere).is_some());

        // Check the distance
        let (distance, _hit_point) =
            ray_sphere_collision(&ray_origin1, &ray_direction1, &sphere).unwrap();
        assert!((distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ray_box3d_collision() {
        let box3d = Box3D {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };

        // Ray pointing directly at the box
        let ray_origin1 = [5.0, 0.0, 0.0];
        let ray_direction1 = [-1.0, 0.0, 0.0];

        // Ray that misses the box
        let ray_origin2 = [5.0, 2.0, 0.0];
        let ray_direction2 = [-1.0, 0.0, 0.0];

        // Ray that starts inside the box
        let ray_origin3 = [0.5, 0.0, 0.0];
        let ray_direction3 = [1.0, 0.0, 0.0];

        assert!(ray_box3d_collision(&ray_origin1, &ray_direction1, &box3d).is_some());
        assert!(ray_box3d_collision(&ray_origin2, &ray_direction2, &box3d).is_none());
        assert!(ray_box3d_collision(&ray_origin3, &ray_direction3, &box3d).is_some());

        // Check the distance
        let (distance, _exit_dist, _hit_point) =
            ray_box3d_collision(&ray_origin1, &ray_direction1, &box3d).unwrap();
        assert!((distance - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_sphere_sphere_collision() {
        let sphere1 = Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };

        let sphere1_velocity = [1.0, 0.0, 0.0];

        let sphere2 = Sphere {
            center: [5.0, 0.0, 0.0],
            radius: 1.0,
        };

        let sphere2_velocity = [-1.0, 0.0, 0.0];

        // The spheres should collide at t = 1.5
        let collision_time = continuous_sphere_sphere_collision(
            &sphere1,
            &sphere1_velocity,
            &sphere2,
            &sphere2_velocity,
            5.0,
        );

        assert!(collision_time.is_some());
        let (time, _pos1, _pos2) = collision_time.unwrap();
        assert!((time - 1.5).abs() < 1e-10);

        // Now with a time step that's too short
        let collision_time = continuous_sphere_sphere_collision(
            &sphere1,
            &sphere1_velocity,
            &sphere2,
            &sphere2_velocity,
            1.0,
        );

        assert!(collision_time.is_none());
    }
}
