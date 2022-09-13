use std::{borrow::Cow, ffi::CStr, ptr};

use ash::{
    extensions::ext::DebugUtils,
    vk::{self},
    Entry,
};
pub use ash::{Device, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{self, Window, WindowBuilder},
};

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
    debug_utils_loader: Option<DebugUtils>,
    debug_callback: Option<vk::DebugUtilsMessengerEXT>,
}

// from: https://github.com/ash-rs/ash/blob/master/examples/src/lib.rs
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
            debug_utils_loader: None,
            debug_callback: None,
        }
    }

    pub fn run(&mut self) {
        let (event_loop, window) = Self::init_window();
        self.init_vulkan(&window);

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

    // Todo: add checks for validaiton layers.
    fn has_validation_layer_support() -> bool {
        true
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
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

            let mut extension_names = ash_window::enumerate_required_extensions(window)
                .unwrap()
                .to_vec();
            extension_names.push(DebugUtils::name().as_ptr());
            extension_names.push(CStr::from_bytes_with_nul_unchecked(
                b"VK_EXT_debug_utils\0",
            ).as_ptr());

            let mut layer_names: Vec<&CStr> = Vec::new();
            if VALIDATION_LAYERS_ENABLED && Self::has_validation_layer_support() {
                println!("VALIDATION LAYERS ACTIVE");
                layer_names.push(CStr::from_bytes_with_nul_unchecked(
                    b"VK_LAYER_KHRONOS_validation\0",
                ));
                
            }

            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                pp_enabled_layer_names: layers_names_raw.as_ptr(),
                pp_enabled_extension_names: extension_names.as_ptr(),
                enabled_layer_count: layer_names.len() as u32,
                enabled_extension_count: extension_names.len() as u32,
                flags: vk::InstanceCreateFlags::default(),
                ..Default::default()
            };
            let instance = entry.create_instance(&create_info, None).unwrap();

            instance
        }
    }

    fn init_vulkan(&mut self, window: &Window) {
        self.entry = Some(Entry::linked());
        self.instance = Some(Self::create_instance(&self.entry.as_ref().unwrap(), window));

        if !VALIDATION_LAYERS_ENABLED {
            return;
        }
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(vulkan_debug_callback),
            p_user_data: ptr::null_mut(),
            ..Default::default()
        };

        assert!(self.entry.is_some());
        assert!(self.instance.is_some());

        self.debug_utils_loader = Some(DebugUtils::new(
            self.entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        ));

        assert!(self.debug_utils_loader.is_some());

        self.debug_callback = unsafe {
            Some(
                self.debug_utils_loader
                    .as_ref()
                    .unwrap()
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap(),
            )
        };
    }

    fn render() {}
    pub fn shutdown(&self) {
        println!("Shutdown called");
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

impl Drop for App {
    fn drop(&mut self) {
        self.shutdown();
    }
}
