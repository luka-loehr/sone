//! Command line arguments: positional words, `--name value` and `--flag`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};

pub struct Args {
    pub words: Vec<String>,
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    /// `flags` are the options that take no value; every other `--name` takes one.
    pub fn parse(args: impl Iterator<Item = String>, flags: &[&str], known: &[&str]) -> Result<Self> {
        let mut out = Args { words: Vec::new(), values: HashMap::new(), flags: Vec::new() };
        let mut args = args;
        while let Some(a) = args.next() {
            match a.strip_prefix("--") {
                Some(name) if flags.contains(&name) => out.flags.push(name.into()),
                Some(name) if known.contains(&name) => {
                    out.values.insert(name.into(), args.next().with_context(|| format!("--{name} needs a value"))?);
                }
                Some(name) => bail!("unknown option --{name}"),
                None => out.words.push(a),
            }
        }
        Ok(out)
    }

    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    pub fn path(&self, name: &str) -> Result<PathBuf> {
        self.values.get(name).map(PathBuf::from).with_context(|| format!("--{name} <path> is missing"))
    }

    pub fn path_or(&self, name: &str, default: &str) -> PathBuf {
        self.values.get(name).map_or_else(|| PathBuf::from(default), PathBuf::from)
    }

    pub fn opt_path(&self, name: &str) -> Option<PathBuf> {
        self.values.get(name).map(PathBuf::from)
    }

    pub fn get<T: FromStr>(&self, name: &str) -> Result<Option<T>>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        self.values.get(name).map(|v| v.parse::<T>().with_context(|| format!("--{name} {v}"))).transpose()
    }

    pub fn get_or<T: FromStr>(&self, name: &str, default: T) -> Result<T>
    where
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        Ok(self.get(name)?.unwrap_or(default))
    }
}
