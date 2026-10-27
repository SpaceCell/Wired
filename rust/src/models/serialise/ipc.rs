// Copyright Peter G. Bower 2025-2026.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Arrow IPC serialisation for minarrow value types.
//!
//! [`Table`] provides the base implementation. Other value types are converted
//! to a table-compatible representation before encoding and converted back
//! after decoding.
//!
//! [`SuperTable`] and [`SuperArray`] use the streaming codec API to encode and
//! decode multiple batches in one IPC stream.
//!
//! [`IpcSerialise`] can be used as a generic bound for types that support this
//! Arrow IPC round trip.

use std::sync::Arc;

use minarrow::{
    Array, BooleanArray, FieldArray, FloatArray, IntegerArray, NumericArray, StringArray,
    SuperArray, SuperTable, Table, TextArray, Vec64,
};

use crate::enums::IPCMessageProtocol;
use crate::error::IoError;
use crate::models::codecs::ipc::ArrowIpcCodec;
use crate::traits::decoder::Decoder;
use crate::traits::encoder::Encoder;
use crate::traits::serialise::Serialise;

#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TemporalArray};

/// Marker trait for minarrow values that support Arrow IPC serialisation.
pub trait IpcSerialise: Serialise<ArrowIpcCodec<Vec64<u8>>, Error = IoError> {}

impl<T> IpcSerialise for T where T: Serialise<ArrowIpcCodec<Vec64<u8>>, Error = IoError> {}

/// Creates an Arrow IPC stream codec.
///
/// Encoding requires a schema. During decoding, the schema is read from the
/// stream.
fn fresh_codec(schema: Vec<minarrow::Field>) -> ArrowIpcCodec<Vec64<u8>> {
    ArrowIpcCodec::new(schema, IPCMessageProtocol::Stream, None, None)
}

// =====================================================================
// Table
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for Table {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let schema = self.schema().iter().map(|f| (**f).clone()).collect();
        let mut codec = fresh_codec(schema);
        Ok(Encoder::encode(&mut codec, self)?)
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let mut codec = fresh_codec(Vec::new());
        Ok(Decoder::decode(&mut codec, bytes)?)
    }

    fn decode_owned(bytes: Vec64<u8>) -> Result<Self, IoError> {
        let mut codec = fresh_codec(Vec::new());
        Ok(Decoder::decode_owned(&mut codec, bytes)?)
    }
}

// =====================================================================
// FieldArray
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for FieldArray {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let table = Table::new("field_array", Some(vec![self.clone()]));
        <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&table)
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let mut table = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)?;
        if table.cols.len() != 1 {
            return Err(IoError::InputDataError(format!(
                "Serialise decode for FieldArray expected a single-column payload, got {} columns",
                table.cols.len()
            )));
        }
        Ok(table.cols.remove(0))
    }

    fn decode_owned(bytes: Vec64<u8>) -> Result<Self, IoError> {
        let mut table = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode_owned(bytes)?;
        if table.cols.len() != 1 {
            return Err(IoError::InputDataError(format!(
                "Serialise decode for FieldArray expected a single-column payload, got {} columns",
                table.cols.len()
            )));
        }
        Ok(table.cols.remove(0))
    }
}

// =====================================================================
// Array
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for Array {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let table = Table::new("array", Some(vec![self.clone().fa("col")]));
        <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&table)
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let mut table = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)?;
        if table.cols.len() != 1 {
            return Err(IoError::InputDataError(format!(
                "Serialise decode for Array expected a single-column payload, got {} columns",
                table.cols.len()
            )));
        }
        Ok(table.cols.remove(0).array)
    }

    fn decode_owned(bytes: Vec64<u8>) -> Result<Self, IoError> {
        let mut table = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode_owned(bytes)?;
        if table.cols.len() != 1 {
            return Err(IoError::InputDataError(format!(
                "Serialise decode for Array expected a single-column payload, got {} columns",
                table.cols.len()
            )));
        }
        Ok(table.cols.remove(0).array)
    }
}

// =====================================================================
// Array categories
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for NumericArray {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::NumericArray(self.clone()))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::NumericArray(n) => Ok(n),
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for NumericArray got a different Array category: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for TextArray {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(self.clone()))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TextArray(t) => Ok(t),
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for TextArray got a different Array category: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

#[cfg(feature = "datetime")]
impl Serialise<ArrowIpcCodec<Vec64<u8>>> for TemporalArray {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TemporalArray(self.clone()))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TemporalArray(t) => Ok(t),
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for TemporalArray got a different Array category: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

// =====================================================================
// Boolean arrays
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for BooleanArray<()> {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::BooleanArray(Arc::new(
            self.clone(),
        )))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::BooleanArray(arc) => {
                Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            }
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for BooleanArray got a different Array category: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

// =====================================================================
// Numeric arrays
// =====================================================================

macro_rules! impl_ipc_numeric_array {
    ($array:ty, $variant:ident) => {
        impl Serialise<ArrowIpcCodec<Vec64<u8>>> for $array {
            type Error = IoError;

            fn encode(&self) -> Result<Vec64<u8>, IoError> {
                let inner = NumericArray::$variant(Arc::new(self.clone()));
                <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::NumericArray(inner))
            }

            fn decode(bytes: &[u8]) -> Result<Self, IoError> {
                match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
                    Array::NumericArray(NumericArray::$variant(arc)) => {
                        Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
                    }
                    other => Err(IoError::InputDataError(format!(
                        "Serialise decode for {} got a different variant: {:?}",
                        stringify!($array),
                        other.arrow_type()
                    ))),
                }
            }
        }
    };
}

impl_ipc_numeric_array!(IntegerArray<i32>, Int32);
impl_ipc_numeric_array!(IntegerArray<i64>, Int64);
impl_ipc_numeric_array!(IntegerArray<u32>, UInt32);
impl_ipc_numeric_array!(IntegerArray<u64>, UInt64);
impl_ipc_numeric_array!(FloatArray<f32>, Float32);
impl_ipc_numeric_array!(FloatArray<f64>, Float64);

#[cfg(feature = "extended_numeric_types")]
impl_ipc_numeric_array!(IntegerArray<i8>, Int8);
#[cfg(feature = "extended_numeric_types")]
impl_ipc_numeric_array!(IntegerArray<i16>, Int16);
#[cfg(feature = "extended_numeric_types")]
impl_ipc_numeric_array!(IntegerArray<u8>, UInt8);
#[cfg(feature = "extended_numeric_types")]
impl_ipc_numeric_array!(IntegerArray<u16>, UInt16);

// =====================================================================
// String arrays
// =====================================================================

macro_rules! impl_ipc_text_array {
    ($array:ty, $variant:ident) => {
        impl Serialise<ArrowIpcCodec<Vec64<u8>>> for $array {
            type Error = IoError;

            fn encode(&self) -> Result<Vec64<u8>, IoError> {
                let inner = TextArray::$variant(Arc::new(self.clone()));
                <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(inner))
            }

            fn decode(bytes: &[u8]) -> Result<Self, IoError> {
                match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
                    Array::TextArray(TextArray::$variant(arc)) => {
                        Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
                    }
                    other => Err(IoError::InputDataError(format!(
                        "Serialise decode for {} got a different variant: {:?}",
                        stringify!($array),
                        other.arrow_type()
                    ))),
                }
            }
        }
    };
}

impl_ipc_text_array!(StringArray<u32>, String32);
#[cfg(feature = "large_string")]
impl_ipc_text_array!(StringArray<u64>, String64);

// =====================================================================
// Categorical arrays
// =====================================================================

#[cfg(feature = "default_categorical_8")]
impl Serialise<ArrowIpcCodec<Vec64<u8>>> for minarrow::CategoricalArray<u8> {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let inner = TextArray::Categorical8(Arc::new(self.clone()));
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(inner))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TextArray(TextArray::Categorical8(arc)) => {
                Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            }
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for CategoricalArray<u8> got a different variant: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

#[cfg(feature = "extended_categorical")]
impl Serialise<ArrowIpcCodec<Vec64<u8>>> for minarrow::CategoricalArray<u16> {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let inner = TextArray::Categorical16(Arc::new(self.clone()));
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(inner))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TextArray(TextArray::Categorical16(arc)) => {
                Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            }
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for CategoricalArray<u16> got a different variant: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for minarrow::CategoricalArray<u32> {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let inner = TextArray::Categorical32(Arc::new(self.clone()));
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(inner))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TextArray(TextArray::Categorical32(arc)) => {
                Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            }
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for CategoricalArray<u32> got a different variant: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

#[cfg(feature = "extended_categorical")]
impl Serialise<ArrowIpcCodec<Vec64<u8>>> for minarrow::CategoricalArray<u64> {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let inner = TextArray::Categorical64(Arc::new(self.clone()));
        <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TextArray(inner))
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
            Array::TextArray(TextArray::Categorical64(arc)) => {
                Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            }
            other => Err(IoError::InputDataError(format!(
                "Serialise decode for CategoricalArray<u64> got a different variant: {:?}",
                other.arrow_type()
            ))),
        }
    }
}

// =====================================================================
// Datetime arrays
// =====================================================================

#[cfg(feature = "datetime")]
macro_rules! impl_ipc_temporal_array {
    ($array:ty, $variant:ident) => {
        impl Serialise<ArrowIpcCodec<Vec64<u8>>> for $array {
            type Error = IoError;

            fn encode(&self) -> Result<Vec64<u8>, IoError> {
                let inner = TemporalArray::$variant(Arc::new(self.clone()));
                <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&Array::TemporalArray(inner))
            }

            fn decode(bytes: &[u8]) -> Result<Self, IoError> {
                match <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(bytes)? {
                    Array::TemporalArray(TemporalArray::$variant(arc)) => {
                        Ok(Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
                    }
                    other => Err(IoError::InputDataError(format!(
                        "Serialise decode for {} got a different variant: {:?}",
                        stringify!($array),
                        other.arrow_type()
                    ))),
                }
            }
        }
    };
}

#[cfg(feature = "datetime")]
impl_ipc_temporal_array!(DatetimeArray<i32>, Datetime32);
#[cfg(feature = "datetime")]
impl_ipc_temporal_array!(DatetimeArray<i64>, Datetime64);

// =====================================================================
// SuperTable
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for SuperTable {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        let schema = self.schema().iter().map(|f| (**f).clone()).collect();
        let mut codec = fresh_codec(schema);
        let mut out: Vec64<u8> = Vec64::new();
        for batch in self.batches() {
            let view: minarrow::TableV = batch.clone().into();
            codec.encode_stream_batch(&view, &mut out, 0, None)?;
        }
        codec.finish(&mut out)?;
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        Self::decode_owned(Vec64::from_slice(bytes))
    }

    fn decode_owned(bytes: Vec64<u8>) -> Result<Self, IoError> {
        let mut codec = fresh_codec(Vec::new());
        let tables = codec.decode_stream(bytes)?;
        let batches: Vec<Arc<Table>> = tables.into_iter().map(Arc::new).collect();
        Ok(SuperTable::from_batches(
            batches,
            Some("super_table".into()),
        ))
    }
}

// =====================================================================
// SuperArray
// =====================================================================

impl Serialise<ArrowIpcCodec<Vec64<u8>>> for SuperArray {
    type Error = IoError;

    fn encode(&self) -> Result<Vec64<u8>, IoError> {
        // Derive the stream schema from the first chunk. Empty values produce
        // an IPC stream without record batches.
        let first_field_array: Option<FieldArray> = self.chunks().first().map(|arr| {
            arr.clone().fa(self
                .field()
                .map(|f| f.name.clone())
                .unwrap_or_else(|| "col".into()))
        });
        let schema: Vec<minarrow::Field> = match &first_field_array {
            Some(fa) => vec![(*fa.field).clone()],
            None => Vec::new(),
        };
        let mut codec = fresh_codec(schema);
        let mut out: Vec64<u8> = Vec64::new();
        let name = self
            .field()
            .map(|f| f.name.clone())
            .unwrap_or_else(|| "col".into());
        for chunk in self.chunks() {
            let table = Table::new(
                "super_array",
                Some(vec![chunk.clone().fa(name.clone())]),
            );
            let view: minarrow::TableV = table.into();
            codec.encode_stream_batch(&view, &mut out, 0, None)?;
        }
        codec.finish(&mut out)?;
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        Self::decode_owned(Vec64::from_slice(bytes))
    }

    fn decode_owned(bytes: Vec64<u8>) -> Result<Self, IoError> {
        let mut codec = fresh_codec(Vec::new());
        let tables = codec.decode_stream(bytes)?;
        let mut chunks: Vec<FieldArray> = Vec::with_capacity(tables.len());
        for mut t in tables {
            if t.cols.len() != 1 {
                return Err(IoError::InputDataError(format!(
                    "Serialise decode for SuperArray expected single-column batches, got {} columns",
                    t.cols.len()
                )));
            }
            chunks.push(t.cols.remove(0));
        }
        Ok(SuperArray::from_field_array_chunks(chunks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minarrow::{
        Array, Buffer, FieldArray, NumericArray, Table, arr_bool, arr_f64, arr_i32, arr_str32,
        vec64,
    };

    fn sample_table() -> Table {
        let ids = FieldArray::from_arr("ids", arr_i32![1, 2, 3, 4]);
        let vals = FieldArray::from_arr("vals", arr_f64![0.5, 1.5, 2.5, 3.5]);
        let labels = FieldArray::from_arr("labels", arr_str32!["a", "b", "c", "d"]);
        Table::new("t", Some(vec![ids, vals, labels]))
    }

    #[test]
    fn table_round_trip() {
        let t = sample_table();
        let bytes = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&t).unwrap();
        let back = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        assert_eq!(back.n_rows, t.n_rows);
        assert_eq!(back.cols.len(), t.cols.len());
        for (orig, got) in t.cols.iter().zip(back.cols.iter()) {
            assert_eq!(orig.field.name, got.field.name);
            assert_eq!(orig.field.dtype, got.field.dtype);
        }
    }

    #[test]
    fn array_round_trip_numeric() {
        let arr = Array::NumericArray(NumericArray::Int32(
            IntegerArray {
                data: Buffer::from(vec64![10, 20, 30]),
                null_mask: None,
            }
            .into(),
        ));
        let bytes = <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&arr).unwrap();
        let back = <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        match (&arr, &back) {
            (
                Array::NumericArray(NumericArray::Int32(a)),
                Array::NumericArray(NumericArray::Int32(b)),
            ) => assert_eq!(a.data.as_ref(), b.data.as_ref()),
            _ => panic!("type changed across round-trip"),
        }
    }

    #[test]
    fn field_array_round_trip() {
        let fa = FieldArray::from_arr("nums", arr_i32![7, 8, 9]);
        let bytes = <FieldArray as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&fa).unwrap();
        let back = <FieldArray as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        assert_eq!(back.field.name, fa.field.name);
        assert_eq!(back.field.dtype, fa.field.dtype);
        assert_eq!(back.len(), fa.len());
    }

    #[test]
    fn integer_array_concrete_round_trip() {
        let ia: IntegerArray<i64> = IntegerArray {
            data: Buffer::from(vec64![100_i64, 200, 300]),
            null_mask: None,
        };
        let bytes =
            <IntegerArray<i64> as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&ia).unwrap();
        let back =
            <IntegerArray<i64> as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        assert_eq!(ia.data.as_ref(), back.data.as_ref());
    }

    #[test]
    fn boolean_array_round_trip() {
        let arr = arr_bool![true, false, true, true];
        let bytes = <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&arr).unwrap();
        let back = <Array as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        match (&arr, &back) {
            (Array::BooleanArray(a), Array::BooleanArray(b)) => assert_eq!(a.len(), b.len()),
            _ => panic!("type changed across round-trip"),
        }
    }

    #[test]
    fn empty_table_round_trip() {
        let t = Table::new(
            "empty",
            Some(vec![FieldArray::from_arr("c", arr_i32![])]),
        );
        let bytes = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&t).unwrap();
        let back = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        assert_eq!(back.n_rows, 0);
        assert_eq!(back.cols.len(), 1);
    }

    #[test]
    fn decode_owned_skips_copy() {
        let t = sample_table();
        let bytes = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&t).unwrap();
        let back = <Table as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode_owned(bytes).unwrap();
        assert_eq!(back.n_rows, t.n_rows);
        assert_eq!(back.cols.len(), t.cols.len());
    }

    #[test]
    fn super_table_round_trip() {
        let b1 = Arc::new(sample_table());
        let b2 = Arc::new(sample_table());
        let st = SuperTable::from_batches(vec![b1, b2], Some("super".into()));
        let bytes = <SuperTable as Serialise<ArrowIpcCodec<Vec64<u8>>>>::encode(&st).unwrap();
        let back = <SuperTable as Serialise<ArrowIpcCodec<Vec64<u8>>>>::decode(&bytes).unwrap();
        assert_eq!(back.n_batches(), 2);
        assert_eq!(back.n_rows(), st.n_rows());
    }
}
