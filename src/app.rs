use core::ffi::c_void;
use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
    path::Path,
    ptr,
};

use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk::{
        self, ColorComponentFlags, CullModeFlags, FrontFace, PolygonMode, PrimitiveTopology,
        SampleCountFlags, ShaderStageFlags,
    },
    Entry,
};
pub use ash::{Device, Instance};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::windows::WindowExtWindows,
    window::{Window, WindowBuilder},
};

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(target_os = "windows")]
use winapi::um::libloaderapi::GetModuleHandleW;

use crate::utils::{self, *};

const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

const REQUIRED_VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

//Todo: find a way to have one of the two, not both
const REQUIRED_DEVICE_EXTENSIONS_RAW: [*const i8; 1] =
    [ash::extensions::khr::Swapchain::name().as_ptr()];
const REQUIRED_DEVICE_EXTENSIONS: [&str; 1] = ["VK_KHR_swapchain"];

struct QueueFamiliyIndices {
    graphics_family: Option<u32>,
    present_familiy: Option<u32>,
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

pub struct App {
    entry: Option<Entry>,
    instance: Option<Instance>,
    device: Option<ash::Device>,
    physical_device: Option<vk::PhysicalDevice>,
    graphics_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,
    surface: Option<vk::SurfaceKHR>,
    surface_loader: Option<ash::extensions::khr::Surface>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_loader: Option<Swapchain>,
    swapchain_images: Option<Vec<vk::Image>>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    swapchain_format: Option<vk::Format>,
    swapchain_extent: Option<vk::Extent2D>,
    swapchain_frame_buffers: Option<Vec<vk::Framebuffer>>,
    render_pass: Option<vk::RenderPass>,
    pipeline_layout: Option<vk::PipelineLayout>,
    graphics_pipeline: Option<vk::Pipeline>,
    command_pool: Option<vk::CommandPool>,
    command_buffers: Option<Vec<vk::CommandBuffer>>,
    image_available_semaphores: Option<Vec<vk::Semaphore>>,
    render_finished_semaphores: Option<Vec<vk::Semaphore>>,
    in_flight_fences: Option<Vec<vk::Fence>>,
    debug_utils_loader: Option<DebugUtils>,
    debug_callback: Option<vk::DebugUtilsMessengerEXT>,
    frame_index: usize,
}

impl App {
    pub fn new() -> Self {
        App {
            entry: None,
            instance: None,
            device: None,
            physical_device: None,
            graphics_queue: None,
            present_queue: None,
            surface: None,
            surface_loader: None,
            swapchain: None,
            swapchain_loader: None,
            swapchain_images: None,
            swapchain_image_views: None,
            swapchain_format: None,
            swapchain_extent: None,
            swapchain_frame_buffers: None,
            render_pass: None,
            pipeline_layout: None,
            graphics_pipeline: None,
            command_pool: None,
            command_buffers: None,
            image_available_semaphores: None,
            render_finished_semaphores: None,
            in_flight_fences: None,
            debug_utils_loader: None,
            debug_callback: None,
            frame_index: 0,
        }
    }

    pub fn run(mut self) {
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
                    // We don't render when minimized
                    if window.inner_size().width != 0 && window.inner_size().height != 0 {
                        // Update renderer
                        self.render();
                        window.request_redraw();
                    }
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

    fn has_validation_layer_support(entry: &Entry) -> bool {
        let layer_properties = entry.enumerate_instance_layer_properties().unwrap();

        if layer_properties.is_empty() {
            eprintln!("No layers availible");
            return false;
        }

        // print availble layers
        println!("Availible Layers: ");
        for layer in layer_properties.iter() {
            println!("- {}", raw_string_to_string(&layer.layer_name));
        }

        for required_layer in REQUIRED_VALIDATION_LAYERS.iter() {
            let mut is_found = false;

            for layer in layer_properties.iter() {
                let layer_name = raw_string_to_string(&layer.layer_name);
                if *required_layer == layer_name {
                    is_found = true;
                    break;
                }
            }

            if !is_found {
                return false;
            }
        }

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
            extension_names
                .push(CStr::from_bytes_with_nul_unchecked(b"VK_EXT_debug_utils\0").as_ptr());

            let mut layer_names: Vec<&CStr> = Vec::new();
            if VALIDATION_LAYERS_ENABLED && Self::has_validation_layer_support(entry) {
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

            entry.create_instance(&create_info, None).unwrap()
        }
    }

    fn init_debug_messenger(&mut self) {
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

    #[cfg(target_os = "windows")]
    fn create_surface(
        &mut self,
        window: &Window,
    ) -> (vk::SurfaceKHR, ash::extensions::khr::Surface) {
        // TODO: Windows only atm. Add support for other platforms later.
        unsafe {
            let hwnd = window.hwnd();
            let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
                s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                hinstance,
                hwnd: hwnd as *const c_void,
            };
            let win32_surface_loader = Win32Surface::new(
                self.entry.as_ref().unwrap(),
                self.instance.as_ref().unwrap(),
            );
            (
                win32_surface_loader
                    .create_win32_surface(&win32_create_info, None)
                    .unwrap(),
                ash::extensions::khr::Surface::new(
                    self.entry.as_ref().unwrap(),
                    self.instance.as_ref().unwrap(),
                ),
            )
        }
    }

    fn get_physical_device(&mut self) -> Option<vk::PhysicalDevice> {
        let devices = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices()
                .unwrap()
        };

        println!("Found {} devices with Vulkan support.", devices.len());
        let mut best_device: Option<vk::PhysicalDevice> = None;

        for device in devices {
            let instance = self.instance.as_ref().unwrap();
            if self.is_physical_device_suitable(instance, &device) {
                let device_properties = unsafe { instance.get_physical_device_properties(device) };
                println!(
                    "{:?} is suitable.",
                    raw_string_to_string(&device_properties.device_name)
                );
                best_device = Some(device);
            }
        }

        best_device
    }

    fn is_physical_device_suitable(
        &self,
        instance: &Instance,
        device: &vk::PhysicalDevice,
    ) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
        let device_features = unsafe { instance.get_physical_device_features(*device) };

        let indices: QueueFamiliyIndices = Self::find_queue_families(
            instance,
            device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let extensions_supported = Self::check_device_extension_support(instance, device);

        let mut swap_chain_supported = false;
        if extensions_supported {
            let swap_chain_support = Self::query_swapchain_support(
                device,
                self.surface.as_ref().unwrap(),
                self.surface_loader.as_ref().unwrap(),
            );

            swap_chain_supported = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty()
        }

        device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            && device_features.geometry_shader > 0
            && indices.graphics_family.is_some()
            && extensions_supported
            && swap_chain_supported
    }

    fn find_queue_families(
        instance: &Instance,
        device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> QueueFamiliyIndices {
        let mut indices = QueueFamiliyIndices {
            graphics_family: None,
            present_familiy: None,
        };

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        for (index, family) in (0_u32..).zip(queue_family_properties.iter()) {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(index);
            }

            let is_present_supported = unsafe {
                surface_loader
                    .get_physical_device_surface_support(*device, index, *surface)
                    .unwrap()
            };

            if family.queue_count > 0 && is_present_supported {
                indices.present_familiy = Some(index);
            }

            if !family.queue_flags.is_empty() {
                break;
            }
        }
        indices
    }

    fn check_device_extension_support(instance: &Instance, device: &vk::PhysicalDevice) -> bool {
        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .expect("Error while getting device extension properties")
        };

        let mut found_extensions = vec![];
        for extension in extension_properties.iter() {
            let extension_name = utils::raw_string_to_string(&extension.extension_name);

            if REQUIRED_DEVICE_EXTENSIONS.contains(&extension_name.as_str()) {
                found_extensions.push(extension_name);
            }
        }

        println!("Found required extensions({}):", found_extensions.len());
        for found_names in found_extensions.iter() {
            println!("- {}", found_names);
        }

        found_extensions.len() == REQUIRED_DEVICE_EXTENSIONS.len()
    }

    fn query_swapchain_support(
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &ash::extensions::khr::Surface,
    ) -> SwapChainSupportDetails {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .expect("Failed to query for surface capabilities.");
            let formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .expect("Failed to query for surface formats.");
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn choose_swap_surface_format(
        available_formats: Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats.iter() {
            if format.format == vk::Format::B8G8R8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(
        available_present_modes: Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for present_mode in available_present_modes.iter() {
            if *present_mode == vk::PresentModeKHR::MAILBOX {
                return vk::PresentModeKHR::MAILBOX;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: clamp(
                    WIDTH as u32,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    HEIGHT as u32,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_swap_chain(&mut self, physical_device: &vk::PhysicalDevice) {
        let swap_chain_support = Self::query_swapchain_support(
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let surface_format: vk::SurfaceFormatKHR =
            Self::choose_swap_surface_format(swap_chain_support.formats);
        let present_mode = Self::choose_swap_present_mode(swap_chain_support.present_modes);
        let extent = Self::choose_swap_extent(swap_chain_support.capabilities);

        let image_count = swap_chain_support.capabilities.min_image_count + 1;
        let image_count = if swap_chain_support.capabilities.max_image_count > 0 {
            image_count.min(swap_chain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: *self.surface.as_ref().unwrap(),
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let indices = Self::find_queue_families(
            self.instance.as_ref().unwrap(),
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );
        let queue_family_indices = vec![
            indices.graphics_family.unwrap(),
            indices.present_familiy.unwrap(),
        ];

        if indices.graphics_family != indices.present_familiy {
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        } else {
            create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = ptr::null();
        }

        self.swapchain_loader = Some(ash::extensions::khr::Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
        ));
        self.swapchain = unsafe {
            Some(
                self.swapchain_loader
                    .as_ref()
                    .unwrap()
                    .create_swapchain(&create_info, None)
                    .expect("Failed to create Swapchain!"),
            )
        };

        let present_images = unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .get_swapchain_images(*self.swapchain.as_ref().unwrap())
                .unwrap()
        };

        self.swapchain_images = Some(Vec::new());
        self.swapchain_images
            .as_mut()
            .unwrap()
            .clone_from(&present_images);

        self.swapchain_extent = Some(extent);
        self.swapchain_format = Some(surface_format.format);
    }

    fn create_image_views(&mut self) {
        self.swapchain_image_views = Some(Vec::new());
        self.swapchain_image_views
            .as_mut()
            .unwrap()
            .reserve(self.swapchain_images.as_ref().unwrap().len());

        for image in self.swapchain_images.as_ref().unwrap() {
            let create_info = vk::ImageViewCreateInfo {
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: *self.swapchain_format.as_ref().unwrap(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            let image_view = unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_image_view(&create_info, None)
                    .expect("Failed to create image views!")
            };
            self.swapchain_image_views
                .as_mut()
                .unwrap()
                .push(image_view);
        }
    }

    fn create_shader_module(&mut self, code: &Vec<u8>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_shader_module(&create_info, None)
                .expect("Failed to create Shader Module!")
        }
    }

    fn create_render_pass(&mut self) {
        let color_attachment = vk::AttachmentDescription {
            format: *self.swapchain_format.as_ref().unwrap(),
            samples: SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let dependency = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let render_pass_attachments = [color_attachment];

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: dependency.as_ptr(),
            ..Default::default()
        };

        self.render_pass = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_render_pass(&render_pass_info, None)
                    .expect("Failed to create render pass!"),
            )
        }
    }

    fn create_graphics_pipeline(&mut self) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_code = read_shader_file(Path::new("assets/shaders/vert.spv"))
            .expect("Error while reading vertex shader");
        let fragment_shader_code = read_shader_file(Path::new("assets/shaders/frag.spv"))
            .expect("Error while reading fragment shader");

        let vertex_shader_module = self.create_shader_module(&vertex_shader_code);
        let fragment_shader_module = self.create_shader_module(&fragment_shader_code);

        let main_function = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::VERTEX,
                module: vertex_shader_module,
                p_name: main_function.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: ShaderStageFlags::FRAGMENT,
                module: fragment_shader_module,
                p_name: main_function.as_ptr(),
                ..Default::default()
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 0,
            vertex_attribute_description_count: 0,
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.as_ref().unwrap().width as f32,
            height: self.swapchain_extent.as_ref().unwrap().height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *self.swapchain_extent.as_ref().unwrap(),
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            ..Default::default()
        };

        let rasterizer_state_create_info = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            ..Default::default()
        }];

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: color_blend_attachments.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterizer_state_create_info,
            //p_multisample_state: TODO
            p_color_blend_state: &color_blending,
            //p_dynamic_state: TODO
            layout: pipeline_layout,
            render_pass: *self.render_pass.as_ref().unwrap(),
            subpass: 0,
            ..Default::default()
        }];

        let graphics_pipelines = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .expect("Failed to create graphics pipeline.")
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(vertex_shader_module, None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(fragment_shader_module, None);
        }

        (graphics_pipelines[0], pipeline_layout)
    }

    fn create_frame_buffers(&mut self) {
        self.swapchain_frame_buffers = Some(Vec::new());
        self.swapchain_frame_buffers
            .as_mut()
            .unwrap()
            .reserve(self.swapchain_image_views.as_ref().unwrap().len());

        for swap_chain_image_view in self.swapchain_image_views.as_ref().unwrap() {
            let framebuffer_info = vk::FramebufferCreateInfo {
                render_pass: *self.render_pass.as_ref().unwrap(),
                attachment_count: 1,
                p_attachments: swap_chain_image_view,
                width: self.swapchain_extent.as_ref().unwrap().width,
                height: self.swapchain_extent.as_ref().unwrap().height,
                layers: 1,
                ..Default::default()
            };

            self.swapchain_frame_buffers.as_mut().unwrap().push(unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_framebuffer(&framebuffer_info, None)
                    .expect("Failed to create framebuffer!")
            })
        }
    }

    fn create_logical_device(
        &mut self,
        physical_device: &vk::PhysicalDevice,
    ) -> QueueFamiliyIndices {
        let indices = Self::find_queue_families(
            self.instance.as_ref().unwrap(),
            physical_device,
            self.surface.as_ref().unwrap(),
            self.surface_loader.as_ref().unwrap(),
        );

        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let physical_device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let requred_validation_layer_raw_names: Vec<&CStr> = REQUIRED_VALIDATION_LAYERS
            .iter()
            .map(|layer_name| unsafe { CStr::from_bytes_with_nul_unchecked(layer_name.as_bytes()) })
            .collect();

        let enable_layer_names: Vec<*const c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: &queue_create_info,
            queue_create_info_count: 1,
            p_enabled_features: &physical_device_features,
            enabled_extension_count: REQUIRED_DEVICE_EXTENSIONS_RAW.len() as u32,
            pp_enabled_extension_names: REQUIRED_DEVICE_EXTENSIONS_RAW.as_ptr(),
            enabled_layer_count: if VALIDATION_LAYERS_ENABLED {
                REQUIRED_VALIDATION_LAYERS.len()
            } else {
                0
            } as u32,
            pp_enabled_layer_names: if VALIDATION_LAYERS_ENABLED {
                enable_layer_names.as_ptr()
            } else {
                ptr::null()
            },
            ..Default::default()
        };

        let device = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .create_device(*physical_device, &create_info, None)
        };
        self.device = Some(device.expect("Error while creating logical device."));

        self.graphics_queue = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.graphics_family.unwrap(), 0)
        });

        self.present_queue = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.present_familiy.unwrap(), 0)
        });

        indices
    }

    fn create_command_pool(&mut self, indices: &QueueFamiliyIndices) {
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: indices.graphics_family.unwrap(),
            ..Default::default()
        };

        self.command_pool = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_command_pool(&create_info, None)
                    .expect("Failed to create command pool!"),
            )
        }
    }

    fn create_command_buffers(&mut self) {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: *self.command_pool.as_ref().unwrap(),
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        self.command_buffers = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to allocate command buffers"),
            )
        }
    }

    fn create_sync_objects(&mut self) {
        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let device = self.device.as_ref().unwrap();

        (
            self.image_available_semaphores,
            self.render_finished_semaphores,
            self.in_flight_fences,
        ) = unsafe {
            let mut image_available_semaphores = Vec::new();
            let mut render_finished_semaphores = Vec::new();
            let mut in_flight_fences = Vec::new();

            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                image_available_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore!"),
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore!"),
                );
                in_flight_fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .expect("Failed to create fence!"),
                );
            }

            (
                Some(image_available_semaphores),
                Some(render_finished_semaphores),
                Some(in_flight_fences),
            )
        };
    }

    fn record_command_buffer(&mut self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo {
            ..Default::default()
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin recording command buffer!");
        }

        let clear_color = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: *self.render_pass.as_ref().unwrap(),
            framebuffer: self.swapchain_frame_buffers.as_ref().unwrap()[image_index],
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: *self.swapchain_extent.as_ref().unwrap(),
            },
            clear_value_count: 1,
            p_clear_values: clear_color.as_ptr(),
            ..Default::default()
        };

        unsafe {
            let device = self.device.as_ref().unwrap();

            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *self.graphics_pipeline.as_ref().unwrap(),
            );

            let _viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain_extent.as_ref().unwrap().width as f32,
                height: self.swapchain_extent.as_ref().unwrap().height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            // Note: Dynamic viewport disabled at this moment.
            //device.cmd_set_viewport(command_buffer, 0, &viewport);

            let _scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: *self.swapchain_extent.as_ref().unwrap(),
            }];
            // Note: Dynamic viewport disabled at this moment.
            //device.cmd_set_scissor(command_buffer, 0, &scissor);

            device.cmd_draw(command_buffer, 3, 1, 0, 0);

            device.cmd_end_render_pass(command_buffer);
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer!");
        }
    }

    fn cleanup_swap_chain(&mut self) {
        let device = self.device.as_ref().unwrap();

        unsafe {
            for frame_buffer in self.swapchain_frame_buffers.as_ref().unwrap() {
                device.destroy_framebuffer(*frame_buffer, None);
            }
            for image_view in self.swapchain_image_views.as_ref().unwrap() {
                device.destroy_image_view(*image_view, None);
            }
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(*self.swapchain.as_ref().unwrap(), None);
        }
    }

    fn recreate_swap_chain(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("Error while waiting for device idle!");
        }

        self.cleanup_swap_chain();

        let physical_device = *self.physical_device.as_ref().unwrap();
        self.create_swap_chain(&physical_device);
        self.create_image_views();
        self.create_frame_buffers();
    }

    fn init_vulkan(&mut self, window: &Window) {
        self.entry = Some(Entry::linked());
        self.instance = Some(Self::create_instance(self.entry.as_ref().unwrap(), window));
        self.init_debug_messenger();
        let (surface, surface_loader) = self.create_surface(window);
        self.surface = Some(surface);
        self.surface_loader = Some(surface_loader);

        self.physical_device = Some(
            self.get_physical_device()
                .expect("Error while getting physical device"),
        );
        let physical_device = *self.physical_device.as_ref().unwrap();

        let indices = self.create_logical_device(&physical_device);
        self.create_swap_chain(&physical_device);
        self.create_image_views();
        self.create_render_pass();
        let (pipeline, layout) = self.create_graphics_pipeline();
        self.graphics_pipeline = Some(pipeline);
        self.pipeline_layout = Some(layout);
        self.create_frame_buffers();
        self.create_command_pool(&indices);
        self.create_command_buffers();
        self.create_sync_objects();
    }

    fn render(&mut self) {
        // Do cool stuff here.
        unsafe {
            let fences = [self.in_flight_fences.as_ref().unwrap()[self.frame_index]];

            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait for Fence!");

            let (image_index, is_sub_optimal) = self
                .swapchain_loader
                .as_ref()
                .unwrap()
                .acquire_next_image(
                    *self.swapchain.as_ref().unwrap(),
                    u64::MAX,
                    self.image_available_semaphores.as_ref().unwrap()[self.frame_index],
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image.");

            if is_sub_optimal {
                return self.recreate_swap_chain();
            }

            let device = self.device.as_ref().unwrap();

            println!("Rendering");

            let wait_semaphores =
                [self.image_available_semaphores.as_ref().unwrap()[self.frame_index]];
            let signal_semaphores =
                [self.render_finished_semaphores.as_ref().unwrap()[self.frame_index]];

            device
                .reset_fences(&fences)
                .expect("Failed to reset fence!");

            device
                .reset_command_buffer(
                    self.command_buffers.as_ref().unwrap()[self.frame_index],
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer!");

            self.record_command_buffer(
                self.command_buffers.as_ref().unwrap()[self.frame_index],
                image_index as usize,
            );

            let submit_info = [vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT].as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: [self.command_buffers.as_ref().unwrap()[self.frame_index]]
                    .as_ptr(),
                signal_semaphore_count: 1,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            }];

            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    *self.graphics_queue.as_ref().unwrap(),
                    &submit_info,
                    self.in_flight_fences.as_ref().unwrap()[self.frame_index],
                )
                .expect("Failed to submit draw command buffer!");

            let swapchains = [*self.swapchain.as_ref().unwrap()];
            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: &image_index,
                ..Default::default()
            };

            let result = self
                .swapchain_loader
                .as_ref()
                .unwrap()
                .queue_present(*self.present_queue.as_ref().unwrap(), &present_info);

            match result {
                Ok(_) => {}
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                        self.recreate_swap_chain()
                    }
                    _ => panic!("Failed to execute queue present."),
                },
            }

            self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    pub fn shutdown(&mut self) {
        println!("Shutdown called");
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("Error while waiting for device idle!");

            self.cleanup_swap_chain();
            let device = self.device.as_ref().unwrap();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                device
                    .destroy_semaphore(self.image_available_semaphores.as_ref().unwrap()[i], None);
                device
                    .destroy_semaphore(self.render_finished_semaphores.as_ref().unwrap()[i], None);
                device.destroy_fence(self.in_flight_fences.as_ref().unwrap()[i], None);
            }

            device.destroy_command_pool(*self.command_pool.as_ref().unwrap(), None);

            device.destroy_pipeline(*self.graphics_pipeline.as_ref().unwrap(), None);
            device.destroy_pipeline_layout(*self.pipeline_layout.as_ref().unwrap(), None);
            device.destroy_render_pass(*self.render_pass.as_ref().unwrap(), None);

            device.destroy_device(None);
            self.surface_loader
                .as_ref()
                .unwrap()
                .destroy_surface(*self.surface.as_ref().unwrap(), None);

            if VALIDATION_LAYERS_ENABLED {
                self.debug_utils_loader
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(*self.debug_callback.as_ref().unwrap(), None);
            }

            self.instance.as_ref().unwrap().destroy_instance(None);
        }
        println!("Shutdown okay");
    }
}

fn clamp(value: u32, min: u32, max: u32) -> u32 {
    let mut value = value;
    if value < min {
        value = min
    }
    if value > max {
        value = max
    }
    value
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
