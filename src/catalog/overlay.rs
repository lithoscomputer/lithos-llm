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

/// Properties the layered catalog loader relies on, for every pair of layers.
#[cfg(test)]
mod properties {
    use proptest::collection::{btree_map, vec};
    use proptest::prelude::*;
    use toml::map::Map;

    use super::{Value, merge};

    fn merged(base: &Value, overlay: &Value) -> Value {
        let mut merged = base.clone();
        merge(&mut merged, overlay.clone());
        merged
    }

    /// Floats are left out: `NaN` is not equal to itself.
    fn leaf() -> impl Strategy<Value = Value> {
        prop_oneof![
            any::<bool>().prop_map(Value::Boolean),
            any::<i64>().prop_map(Value::Integer),
            "[a-z]{0,3}".prop_map(Value::String),
            vec(any::<i64>().prop_map(Value::Integer), 0..3).prop_map(Value::Array),
        ]
    }

    /// Keys come from a three-letter alphabet so layers often collide.
    fn table_of(value: impl Strategy<Value = Value>) -> impl Strategy<Value = Value> {
        btree_map("[a-c]", value, 0..4)
            .prop_map(|entries| Value::Table(entries.into_iter().collect()))
    }

    fn value() -> impl Strategy<Value = Value> {
        leaf().prop_recursive(3, 24, 4, table_of)
    }

    /// A layer, which is always a table at its root.
    fn layer() -> impl Strategy<Value = Value> {
        table_of(value())
    }

    /// Every non-table value in `value` with the key path that reaches it.
    fn leaves(value: &Value) -> Vec<(Vec<String>, &Value)> {
        let Value::Table(table) = value else {
            return vec![(Vec::new(), value)];
        };
        table
            .iter()
            .flat_map(|(key, child)| {
                leaves(child).into_iter().map(|(mut path, leaf)| {
                    path.insert(0, key.clone());
                    (path, leaf)
                })
            })
            .collect()
    }

    fn get<'a>(value: &'a Value, path: &[String]) -> Option<&'a Value> {
        path.iter()
            .try_fold(value, |node, key| node.as_table()?.get(key))
    }

    /// Whether `overlay` sets `path` or replaces one of its ancestors.
    ///
    /// Walking `overlay` along `path` either runs out of keys, leaving the base
    /// value in place, or reaches a value that overwrites it.
    fn overlay_covers(overlay: &Value, path: &[String]) -> bool {
        let mut node = overlay;
        for key in path {
            let Value::Table(table) = node else {
                return true;
            };
            let Some(next) = table.get(key) else {
                return false;
            };
            node = next;
        }
        true
    }

    proptest! {
        #[test]
        fn an_empty_overlay_keeps_the_base(base in layer()) {
            prop_assert_eq!(merged(&base, &Value::Table(Map::new())), base);
        }

        #[test]
        fn an_empty_base_takes_the_overlay(overlay in layer()) {
            prop_assert_eq!(merged(&Value::Table(Map::new()), &overlay), overlay);
        }

        #[test]
        fn merging_the_same_overlay_twice_changes_nothing(base in layer(), overlay in layer()) {
            let once = merged(&base, &overlay);
            prop_assert_eq!(merged(&once, &overlay), once);
        }

        #[test]
        fn every_overlay_value_wins(base in layer(), overlay in layer()) {
            let result = merged(&base, &overlay);
            for (path, value) in leaves(&overlay) {
                prop_assert_eq!(get(&result, &path), Some(value), "at {:?}", path);
            }
        }

        #[test]
        fn base_values_the_overlay_does_not_reach_survive(base in layer(), overlay in layer()) {
            let result = merged(&base, &overlay);
            for (path, value) in leaves(&base) {
                if !overlay_covers(&overlay, &path) {
                    prop_assert_eq!(get(&result, &path), Some(value), "at {:?}", path);
                }
            }
        }

        #[test]
        fn every_merged_value_comes_from_a_layer(base in layer(), overlay in layer()) {
            for (path, value) in leaves(&merged(&base, &overlay)) {
                prop_assert!(
                    get(&overlay, &path) == Some(value) || get(&base, &path) == Some(value),
                    "{:?} at {:?} is in neither layer",
                    value,
                    path,
                );
            }
        }
    }
}
