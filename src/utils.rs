use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::os::raw::c_char;
use std::path::Path;

use ash::vk;

// Adapted from: https://github.com/unknownue/vulkan-tutorial-rust/blob/master/src/utility/tools.rs
pub fn raw_string_to_string(raw_string: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Error while converting vulkan raw string")
        .to_owned()
}

/// # Safety
///
// from: https://github.com/ash-rs/ash/blob/master/examples/src/lib.rs
pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number;

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
        "[{:?} - {:?}] ({}({})) : {}",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

pub fn read_shader_file(path: &Path) -> Result<Vec<u8>, String> {
    if path.exists() {
        let spv_file = File::open(path).expect("Unexpected error while reading shader file.");
        let contents: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();
        return Ok(contents);
    }
    Err(format!("Path: {:?} does not exist..", path))
}
