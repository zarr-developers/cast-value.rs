//! Check what SIMD instructions pulp::Arch actually dispatches to on this CPU.

use pulp::{Arch, Simd, WithSimd};

struct InspectKernel;

impl WithSimd for InspectKernel {
    type Output = ();

    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
        println!("SIMD type: {}", std::any::type_name::<S>());
        println!("F64_LANES: {}", S::F64_LANES);
        println!("F32_LANES: {}", S::F32_LANES);
    }
}

fn main() {
    println!("CPU: AMD FX-8350 (Piledriver, 2012)");
    println!("Available: AVX, FMA4, XOP, SSE4.2, SSE4.1, SSSE3");
    println!("NOT available: AVX2, FMA3\n");

    println!("pulp::Arch::new().dispatch() uses:");
    Arch::new().dispatch(InspectKernel);
}
