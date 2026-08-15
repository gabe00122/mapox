use crate::vocab::{VocabId, Vocabulary};

pub trait VocabEnum: Copy + Into<VocabId> {
    fn symbol(self) -> &'static str;
    fn vocab() -> Vocabulary;
    fn from_id(id: VocabId) -> Self;
}

#[macro_export]
macro_rules! vocab_enum {
    ($vis:vis $name:ident { $($variant:ident => $symbol:expr),+ $(,)? }) => {
        #[repr(u16)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $name {
            $($variant),+
        }

        impl $name {
            const TABLE: &[$name] = &[$($name::$variant),+];
        }

        impl $crate::envs::common::vocab_enum::VocabEnum for $name {
            fn symbol(self) -> &'static str {
                match self {
                    $(Self::$variant => $symbol),+
                }
            }

            fn vocab() -> $crate::vocab::Vocabulary {
                let mut out = Vocabulary::new();
                for s in Self::TABLE.iter() {
                    out.add(s.symbol());
                }
                out
            }

            fn from_id(id: VocabId) -> Self {
                Self::TABLE[id as usize]
            }
        }

        impl ::core::convert::From<$name> for $crate::vocab::VocabId {
            fn from(value: $name) -> Self {
                value as $crate::vocab::VocabId
            }
        }
    };
}
