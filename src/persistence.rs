use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use directories::ProjectDirs;

use crate::{calibration::CalibrationProfile, config::AppConfig};

const APP_QUALIFIER: &str = "org";
const APP_ORGANISATION: &str = "EyeOS";
const APP_NAME: &str = "EyeOS";
const CONFIG_FILE: &str = "settings.json";
const CALIBRATION_FILE: &str = "calibration.dpapi";

#[derive(Debug, Clone)]
pub struct ProfileStore {
    root: PathBuf,
}

impl ProfileStore {
    pub fn for_current_user() -> Result<Self> {
        let directories = ProjectDirs::from(APP_QUALIFIER, APP_ORGANISATION, APP_NAME)
            .ok_or_else(|| anyhow!("Windows did not provide a local application-data directory"))?;
        Ok(Self {
            root: directories.config_local_dir().to_path_buf(),
        })
    }

    pub fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn load_config(&self) -> Result<AppConfig> {
        let path = self.root.join(CONFIG_FILE);
        if !path.exists() {
            return Ok(AppConfig::default());
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let config: AppConfig = serde_json::from_slice(&bytes).context("reading EyeOS settings")?;
        Ok(config.migrate())
    }

    pub fn save_config(&self, config: &AppConfig) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        let bytes = serde_json::to_vec_pretty(config).context("serializing EyeOS settings")?;
        fs::write(self.root.join(CONFIG_FILE), bytes).context("saving EyeOS settings")
    }

    pub fn load_calibration(&self) -> Result<Option<CalibrationProfile>> {
        let path = self.root.join(CALIBRATION_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let protected = fs::read(&path).context("reading encrypted calibration profile")?;
        let raw = dpapi::unprotect(&protected)?;
        let profile = serde_json::from_slice(&raw).context("reading calibration profile")?;
        Ok(Some(profile))
    }

    pub fn save_calibration(&self, calibration: &CalibrationProfile) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("creating {}", self.root.display()))?;
        let raw = serde_json::to_vec(calibration).context("serializing calibration profile")?;
        let protected = dpapi::protect(&raw)?;
        fs::write(self.root.join(CALIBRATION_FILE), protected)
            .context("saving encrypted calibration profile")
    }

    pub fn reset(&self) -> Result<()> {
        for filename in [CONFIG_FILE, CALIBRATION_FILE] {
            let path = self.root.join(filename);
            if path.exists() {
                fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            }
        }
        Ok(())
    }
}

pub fn install_autostart() -> Result<()> {
    #[cfg(windows)]
    {
        autostart::install()
    }
    #[cfg(not(windows))]
    {
        Err(anyhow!("automatic startup is only supported on Windows"))
    }
}

#[cfg(windows)]
mod dpapi {
    use std::{
        ptr::{null, null_mut},
        slice,
    };

    use anyhow::{Result, anyhow};
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::Cryptography::{
            CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData, CryptUnprotectData,
        },
    };

    pub fn protect(bytes: &[u8]) -> Result<Vec<u8>> {
        crypt(bytes, true)
    }

    pub fn unprotect(bytes: &[u8]) -> Result<Vec<u8>> {
        crypt(bytes, false)
    }

    fn crypt(bytes: &[u8], protect: bool) -> Result<Vec<u8>> {
        let input = CRYPT_INTEGER_BLOB {
            cbData: bytes.len() as u32,
            pbData: bytes.as_ptr() as *mut u8,
        };
        let mut output = CRYPT_INTEGER_BLOB {
            cbData: 0,
            pbData: null_mut(),
        };
        let success = unsafe {
            if protect {
                CryptProtectData(
                    &input,
                    null(),
                    null(),
                    null(),
                    null(),
                    CRYPTPROTECT_UI_FORBIDDEN,
                    &mut output,
                )
            } else {
                CryptUnprotectData(
                    &input,
                    null_mut(),
                    null(),
                    null(),
                    null(),
                    CRYPTPROTECT_UI_FORBIDDEN,
                    &mut output,
                )
            }
        };
        if success == 0 {
            return Err(anyhow!(
                "Windows DPAPI could not protect the calibration profile"
            ));
        }
        let result =
            unsafe { slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
        unsafe { LocalFree(output.pbData.cast()) };
        Ok(result)
    }
}

#[cfg(not(windows))]
mod dpapi {
    use anyhow::Result;

    pub fn protect(bytes: &[u8]) -> Result<Vec<u8>> {
        Ok(bytes.to_vec())
    }
    pub fn unprotect(bytes: &[u8]) -> Result<Vec<u8>> {
        Ok(bytes.to_vec())
    }
}

#[cfg(windows)]
mod autostart {
    use std::{env, iter::once, ptr::null_mut};

    use anyhow::{Context, Result, anyhow};
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey,
        RegCreateKeyExW, RegSetValueExW,
    };

    const RUN_KEY: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";

    pub fn install() -> Result<()> {
        let executable = env::current_exe().context("locating eyeos.exe")?;
        let command = format!("\"{}\"", executable.display());
        let key_path = wide(RUN_KEY);
        let value_name = wide("EyeOS");
        let value = wide(&command);
        let mut key: HKEY = null_mut();
        let created = unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                key_path.as_ptr(),
                0,
                null_mut(),
                REG_OPTION_NON_VOLATILE,
                KEY_SET_VALUE,
                null_mut(),
                &mut key,
                null_mut(),
            )
        };
        if created != 0 {
            return Err(anyhow!(
                "Windows could not create the EyeOS startup entry ({created})"
            ));
        }
        let result = unsafe {
            RegSetValueExW(
                key,
                value_name.as_ptr(),
                0,
                REG_SZ,
                value.as_ptr() as *const u8,
                (value.len() * std::mem::size_of::<u16>()) as u32,
            )
        };
        unsafe { RegCloseKey(key) };
        if result != 0 {
            return Err(anyhow!(
                "Windows could not save the EyeOS startup entry ({result})"
            ));
        }
        Ok(())
    }

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(once(0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::CalibrationProfile;

    #[test]
    fn preferences_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let store = ProfileStore::at(directory.path().to_path_buf());
        let config = AppConfig {
            high_contrast: true,
            ..AppConfig::default()
        };
        store.save_config(&config).unwrap();
        assert_eq!(store.load_config().unwrap(), config);
    }

    #[test]
    fn calibration_is_protected_and_round_trips() {
        let directory = tempfile::tempdir().unwrap();
        let store = ProfileStore::at(directory.path().to_path_buf());
        let profile = CalibrationProfile {
            x_coefficients: [1.0, 2.0, 3.0],
            y_coefficients: [4.0, 5.0, 6.0],
            sample_count: 9,
            median_error_px: 1.5,
        };
        store.save_calibration(&profile).unwrap();
        assert_eq!(store.load_calibration().unwrap(), Some(profile));
    }
}
