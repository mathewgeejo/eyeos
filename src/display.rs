//! Read physical geometry from the primary monitor's EDID, never assumed DPI.
use crate::Point;

#[derive(Debug, Clone)]
pub struct DisplayGeometry {
    pub size_mm: Point,
    pub device_id: String,
}

pub fn physical_to_logical(point: Point, pixels_per_point: f32) -> Point {
    let scale = f64::from(pixels_per_point.max(0.1));
    Point::new(point.x / scale, point.y / scale)
}

/// Validate the base block checksum and prefer the native detailed timing's
/// millimetres. EDID's centimetre fields are a lower-resolution fallback.
pub fn edid_size_mm(edid: &[u8]) -> Option<Point> {
    if edid.len() < 128
        || edid[..8] != [0, 255, 255, 255, 255, 255, 255, 0]
        || edid[..128].iter().fold(0u8, |sum, b| sum.wrapping_add(*b)) != 0
    {
        return None;
    }
    let usable = |w: u16, h: u16| {
        (w >= 100 && h >= 60 && w <= 3000 && h <= 2000)
            .then(|| Point::new(f64::from(w), f64::from(h)))
    };
    for d in edid[54..126].chunks_exact(18) {
        if d[0] == 0 && d[1] == 0 {
            continue;
        }
        let w = u16::from(d[12]) | (u16::from(d[14] & 0xf0) << 4);
        let h = u16::from(d[13]) | (u16::from(d[14] & 0x0f) << 8);
        if let Some(size) = usable(w, h) {
            return Some(size);
        }
    }
    usable(u16::from(edid[21]) * 10, u16::from(edid[22]) * 10)
}

#[cfg(windows)]
pub fn primary_display_geometry() -> Option<DisplayGeometry> {
    use windows_sys::Win32::{
        Graphics::Gdi::{
            DISPLAY_DEVICE_ACTIVE, DISPLAY_DEVICE_PRIMARY_DEVICE, DISPLAY_DEVICEW,
            EnumDisplayDevicesW,
        },
        System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_BINARY, RegGetValueW},
    };
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    let string = |v: &[u16]| {
        String::from_utf16_lossy(&v[..v.iter().position(|x| *x == 0).unwrap_or(v.len())])
    };
    for i in 0..32 {
        let mut adapter: DISPLAY_DEVICEW = unsafe { std::mem::zeroed() };
        adapter.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;
        if unsafe { EnumDisplayDevicesW(std::ptr::null(), i, &mut adapter, 0) } == 0 {
            break;
        }
        if adapter.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE == 0 {
            continue;
        }
        for j in 0..16 {
            let mut monitor: DISPLAY_DEVICEW = unsafe { std::mem::zeroed() };
            monitor.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;
            // EDD_GET_DEVICE_INTERFACE_NAME gives the registry instance identity.
            if unsafe { EnumDisplayDevicesW(adapter.DeviceName.as_ptr(), j, &mut monitor, 1) } == 0
            {
                break;
            }
            if monitor.StateFlags & DISPLAY_DEVICE_ACTIVE == 0 {
                continue;
            }
            let id = string(&monitor.DeviceID);
            let parts: Vec<_> = id.split('#').collect();
            if parts.len() < 3 || !parts[0].to_ascii_uppercase().ends_with("DISPLAY") {
                continue;
            }
            let key = wide(&format!(
                "SYSTEM\\CurrentControlSet\\Enum\\DISPLAY\\{}\\{}\\Device Parameters",
                parts[1], parts[2]
            ));
            let value = wide("EDID");
            let mut data = [0u8; 4096];
            let mut length = data.len() as u32;
            if unsafe {
                RegGetValueW(
                    HKEY_LOCAL_MACHINE,
                    key.as_ptr(),
                    value.as_ptr(),
                    RRF_RT_REG_BINARY,
                    std::ptr::null_mut(),
                    data.as_mut_ptr().cast(),
                    &mut length,
                )
            } != 0
            {
                continue;
            }
            if let Some(size_mm) = edid_size_mm(&data[..(length as usize).min(data.len())]) {
                return Some(DisplayGeometry {
                    size_mm,
                    device_id: id,
                });
            }
        }
    }
    None
}
#[cfg(not(windows))]
pub fn primary_display_geometry() -> Option<DisplayGeometry> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> [u8; 128] {
        let mut e = [0; 128];
        e[..8].copy_from_slice(&[0, 255, 255, 255, 255, 255, 255, 0]);
        e[21] = 34;
        e[22] = 19;
        e[54] = 1;
        e[66] = 88;
        e[67] = 193;
        e[68] = 0x10;
        e[127] = 0u8.wrapping_sub(e[..127].iter().fold(0u8, |s, v| s.wrapping_add(*v)));
        e
    }
    #[test]
    fn edid_prefers_actual_mm_and_rejects_corruption_or_unknown_size() {
        let mut e = fixture();
        assert_eq!(edid_size_mm(&e), Some(Point::new(344.0, 193.0)));
        e[66] ^= 1;
        assert_eq!(edid_size_mm(&e), None);
        assert_eq!(edid_size_mm(&[0; 128]), None);
    }
    #[test]
    fn targets_and_gaze_use_same_coordinate_space_at_scaled_dpi() {
        let p = Point::new(960.0, 540.0);
        assert_eq!(physical_to_logical(p, 1.25), Point::new(768.0, 432.0));
        assert_eq!(physical_to_logical(p, 2.0), Point::new(480.0, 270.0));
    }
}
