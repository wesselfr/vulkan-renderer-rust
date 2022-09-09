use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{UserAttentionType, Window, WindowBuilder},
};

pub mod app;
use app::*;

fn main() {
    let mut app = App::new();
    app.run();
}
