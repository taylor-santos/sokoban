use std::ops::Neg;

use cgmath::{Deg, Matrix4, Point3, Rad, Vector3};
use cgmath::num_traits::clamp;
use cgmath::prelude::*;

#[derive(Debug)]
pub struct FirstPersonCamera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub speed: f32,
    pub sensitivity: f32,
}

impl FirstPersonCamera {
    pub fn new() -> FirstPersonCamera {
        FirstPersonCamera {
            position: Point3::new(0.0, 0.0, 0.0),
            yaw: Rad(0.0),
            pitch: Rad(0.0),
            speed: 5.0,
            sensitivity: 0.1,
        }
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw = (self.yaw + Rad::from(Deg(delta_x * self.sensitivity))) % Rad::full_turn();
        let min = Rad::turn_div_4().neg();
        let max = Rad::turn_div_4();
        self.pitch = clamp(self.pitch - Rad::from(Deg(delta_y * self.sensitivity)), min, max);
    }

    pub fn move_forward(&mut self, delta_time: f32) {
        let forward = self.get_forward();
        self.position += forward * self.speed * delta_time;
    }

    pub fn move_backward(&mut self, delta_time: f32) {
        let forward = self.get_forward();
        self.position -= forward * self.speed * delta_time;
    }

    pub fn move_left(&mut self, delta_time: f32) {
        let right = self.get_right();
        self.position -= right * self.speed * delta_time;
    }

    pub fn move_right(&mut self, delta_time: f32) {
        let right = self.get_right();
        self.position += right * self.speed * delta_time;
    }

    fn get_forward(&self) -> Vector3<f32> {
        let pitch_cos = self.pitch.cos();
        Vector3::new(
            -self.yaw.sin() * pitch_cos,
            self.pitch.sin(),
            self.yaw.cos() * pitch_cos,
        )
    }

    fn get_up(&self) -> Vector3<f32> {
        let pitch_sin = self.pitch.sin();
        Vector3::new(
            self.yaw.sin() * pitch_sin,
            self.pitch.cos(),
            -self.yaw.cos() * pitch_sin,
        )
    }

    fn get_right(&self) -> Vector3<f32> {
        Vector3::new(
            -self.yaw.cos(),
            0.0,
            -self.yaw.sin(),
        )
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        let forward = self.get_forward();
        let up = self.get_up();

        Matrix4::look_at_rh(self.position, self.position + forward, up)
    }
}
