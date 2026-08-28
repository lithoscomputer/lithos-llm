use toml::Value;

pub(crate) fn merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base), Value::Table(overlay)) => {
            for (key, value) in overlay {
                if let Some(existing) = base.get_mut(&key) {
                    merge(existing, value);
                } else {
                    base.insert(key, value);
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

#[cfg(test)]
mod tests {
    use toml::de::Error as TomlError;

    use super::{Value, merge};

    #[test]
    fn recursively_merges_tables_and_replaces_arrays() -> Result<(), TomlError> {
        let mut base = toml::from_str::<Value>(
            r"
            values = [1, 2]
            [metadata.app]
            first = true
            nested = { left = 1 }
            ",
        )?;
        let overlay = toml::from_str::<Value>(
            r"
            values = [3]
            [metadata.app]
            second = true
            nested = { right = 2 }
            ",
        )?;

        merge(&mut base, overlay);

        assert_eq!(base["values"].as_array().map(Vec::len), Some(1));
        assert_eq!(base["metadata"]["app"]["first"].as_bool(), Some(true));
        assert_eq!(base["metadata"]["app"]["second"].as_bool(), Some(true));
        assert_eq!(
            base["metadata"]["app"]["nested"]["left"].as_integer(),
            Some(1)
        );
        assert_eq!(
            base["metadata"]["app"]["nested"]["right"].as_integer(),
            Some(2)
        );
        Ok(())
    }
}
