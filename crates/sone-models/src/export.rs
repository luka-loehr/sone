//! Tensors out of Burn under `sone-infer`'s names. Burn's `Linear` stores its
//! weight `(in, out)`; the files use the PyTorch layout `(out, in)`.

use burn::nn::Linear;
use burn::prelude::*;
use sone_infer::st::NamedTensor;

use crate::layers::{Attention, FeedForward, TimeBlock};

pub fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().expect("f32 data")
}

pub fn t1<B: Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    values(t)
}

pub fn t2<B: Backend>(t: Tensor<B, 2>) -> (Vec<f32>, [usize; 2]) {
    let dims = t.dims();
    (values(t), dims)
}

/// Row `i` of a matrix from [`t2`].
pub fn row(m: &(Vec<f32>, [usize; 2]), i: usize) -> Vec<f32> {
    m.0[i * m.1[1]..(i + 1) * m.1[1]].to_vec()
}

/// An `(in, out)` matrix written as `(out, in)`.
pub fn transposed<B: Backend>(name: &str, w: Tensor<B, 2>) -> NamedTensor {
    let [i, o] = w.dims();
    (name.to_string(), vec![o, i], values(w.transpose()))
}

pub fn lin<B: Backend>(out: &mut Vec<NamedTensor>, p: &str, l: &Linear<B>) {
    out.push(transposed(&format!("{p}.w"), l.weight.val()));
    if let Some(b) = &l.bias {
        let n = b.val().dims()[0];
        out.push((format!("{p}.b"), vec![n], t1(b.val())));
    }
}

pub fn feed_forward<B: Backend>(out: &mut Vec<NamedTensor>, p: &str, ff: &FeedForward<B>) {
    let g = ff.norm.gain.val();
    out.push((format!("{p}.norm"), g.dims().to_vec(), t1(g)));
    lin(out, &format!("{p}.l1"), &ff.l1);
    lin(out, &format!("{p}.l2"), &ff.l2);
}

pub fn attention<B: Backend>(out: &mut Vec<NamedTensor>, p: &str, a: &Attention<B>) {
    let g = a.norm.gain.val();
    out.push((format!("{p}.norm"), g.dims().to_vec(), t1(g)));
    out.push(transposed(&format!("{p}.qkv"), a.qkv.weight.val()));
    lin(out, &format!("{p}.out"), &a.out);
}

pub fn time_block<B: Backend>(out: &mut Vec<NamedTensor>, p: &str, tb: &TimeBlock<B>) {
    let g = tb.norm.gain.val();
    out.push((format!("{p}.norm"), g.dims().to_vec(), t1(g)));
    let (w, dims) = t2(tb.conv_w.val());
    out.push((format!("{p}.conv.w"), dims.to_vec(), w));
    let b = tb.conv_b.val();
    out.push((format!("{p}.conv.b"), b.dims().to_vec(), t1(b)));
    lin(out, &format!("{p}.l1"), &tb.l1);
    lin(out, &format!("{p}.l2"), &tb.l2);
}

/// A complete safetensors file.
pub fn to_safetensors(model: &str, config: &impl serde::Serialize, tensors: &[NamedTensor], half: bool) -> anyhow::Result<Vec<u8>> {
    sone_infer::st::write(tensors, half, &sone_infer::st::metadata(model, config)?)
}

/// Checks that exported names and shapes are exactly the reader's list.
pub fn check_names(tensors: &[NamedTensor], shapes: &sone_infer::config::Shapes) -> anyhow::Result<()> {
    let mut got: Vec<(String, Vec<usize>)> = tensors.iter().map(|(n, s, _)| (n.clone(), s.clone())).collect();
    let mut want = shapes.clone();
    got.sort();
    want.sort();
    if got != want {
        let missing: Vec<_> = want.iter().filter(|w| !got.contains(w)).take(5).collect();
        let extra: Vec<_> = got.iter().filter(|g| !want.contains(g)).take(5).collect();
        anyhow::bail!("export names differ: missing {missing:?}, extra {extra:?}");
    }
    Ok(())
}
