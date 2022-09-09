use std::ffi::CStr;

use ash::{vk, Entry};
pub use ash::{Device, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
        }
    }

    pub fn run(&mut self) {
        let (event_loop, window) = Self::init_window();
        self.init_vulkan();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => {
                    // Update renderer
                    Self::render();
                    window.request_redraw();
                }
                _ => (),
            }
        });

        self.shutdown();
    }

    fn init_window() -> (EventLoop<()>, Window) {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_resizable(true)
            .with_title("Vulkan Renderer")
            .build(&event_loop)
            .unwrap();
        (event_loop, window)
    }

    fn create_instance(entry: &Entry) -> Instance {
        unsafe {
            let app_name = CStr::from_bytes_with_nul_unchecked(b"VulkanTriangle\0");
            let engine_name = CStr::from_bytes_with_nul_unchecked(b"VulkanRenderer\0");

            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: 0,
                p_engine_name: engine_name.as_ptr(),
                engine_version: 0,
                api_version: vk::make_api_version(0, 1, 0, 0),
                ..Default::default()
            };

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            };
            let instance = entry.create_instance(&create_info, None).unwrap();

            instance
        }
    }

    fn init_vulkan(&mut self) {
        self.entry = Some(unsafe { Entry::load().expect("Error while loading vulkan") });
        self.instance = Some(Self::create_instance(&self.entry.as_ref().unwrap()));
    }

    fn render() {}
    fn shutdown(&self) {
        unsafe {
            self.instance.as_ref().unwrap().destroy_instance(None);
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
