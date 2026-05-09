# SKILL: Candle ML Framework (Rust)

## 1. Core Abstractions

### Philosophy and Structure
Candle represents a fundamental departure from PyTorch's C++ backend approach by leveraging Rust's ownership system for memory safety and zero-cost abstractions. The framework is built around three core concepts:

**Tensor:**
- Reference-counted via `Arc<Tensor_>` enabling cheap clones without data duplication
- Contains `Storage` (device-specific memory), `Layout` (stride/offset/shape), `BackpropOp` (computational graph), `DType`, and `Device`
- Storage is wrapped in `Arc<RwLock<Storage>>` for thread-safe mutable access
- Unlike PyTorch, tensors can have non-contiguous memory layouts without copying

```rust
// Core tensor structure (simplified)
pub struct Tensor(Arc<Tensor_>);
pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>,  // Device-specific memory
    layout: Layout,                  // Stride, offset, shape
    op: BackpropOp,                  // For autograd
    dtype: DType,
    device: Device,
}
```

**Device:**
- Enum abstraction over `Cpu`, `Cuda`, and `Metal` backends
- Each device implements `BackendDevice` trait for memory allocation, synchronization, and operations
- Devices are lightweight and can be cloned; actual GPU contexts are shared via `Arc`
- Supports location-based device matching and per-device stream management

```rust
pub enum Device {
    Cpu,
    Cuda(CudaDevice),    // Wraps cudarc::driver::CudaContext
    Metal(MetalDevice),
}

// Backend device trait
pub trait BackendDevice: Clone + Send + Sync {
    type Storage: BackendStorage;
    fn new(_: usize) -> Result<Self>;
    fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    fn synchronize(&self) -> Result<()>;  // Block until all ops complete
}
```

**DType:**
- Comprehensive type system including float (F16, BF16, F32, F64), integer (I16, I32, I64, U8, U32)
- Specialized quantization types (F8E4M3, F6E2M3, F6E3M2, F4) for efficient memory usage
- GGUF quantization types (Q4_0, Q4_1, Q5_0, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K) for compressed models
- Type conversions are explicit and handled by backend implementations

### Key Differences from PyTorch

**Memory Management:**
- PyTorch: Python ref counting + C++ allocator, manual memory management
- Candle: Rust ownership + Arc ref counting, compile-time memory safety
- PyTorch: Eager execution with optional graph mode
- Candle: Always builds computational graph (via BackpropOp), but lazy on GPU

**Error Handling:**
- PyTorch: Python exceptions, C++ assertions
- Candle: `Result<T>` with context chaining, compile-time error handling

**Type Safety:**
- PyTorch: Dynamic typing, runtime checks
- Candle: Static typing, compile-time device/dtype validation

## 2. Architectural & Memory Patterns

### Memory Safety in Rust Context

**Thread Safety:**
- Tensors are `Send + Sync` due to `Arc<RwLock<Storage>>` wrapper
- Operations acquire read/write locks on storage, preventing data races
- GPU operations are enqueued on device-specific streams, executed asynchronously
- Multiple threads can safely clone tensors, but concurrent writes to same storage are prevented by RwLock

**Zero-Copy Operations:**
- Reshape, transpose, and slice operations create new Tensors with same `Arc<Storage>`
- Only `Layout` changes (stride, offset, shape), no memory allocation
- Enables efficient view-based computation without data copying

```rust
// Reshape is zero-copy - only layout changes
let reshaped = tensor.reshape((batch_size, seq_len, hidden_dim))?;

// Slice creates view into existing storage
let sliced = tensor.narrow(0, 0, batch_size)?;
```

**Memory Pooling:**
- CUDA backends use cudarc's allocator with automatic memory pooling
- Metal backends similar approach
- CPU uses standard Rust allocator with potential for custom allocators

### Automatic Differentiation (Autograd)

**Computational Graph:**
- Each tensor carries optional `BackpropOp` describing how it was created
- Graph is built dynamically during forward pass
- Backward pass uses topological sort of nodes, computing gradients recursively

**Gradient Tracking:**
- Only tensors created with gradient requirements track operations
- In-place operations break gradient computation
- Detaching stops gradient propagation (creates tensor with `BackpropOp::none()`)

```rust
// Gradient computation (simplified)
impl Tensor {
    pub fn backward(&self, grad: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let nodes = self.sorted_nodes();  // Topological sort
        let mut grads = HashMap::new();
        grads.insert(self.id(), grad.clone());
        
        for node in nodes.iter().rev() {
            if let Some(op) = node.op() {
                op.backward(node, &grads)?;
            }
        }
        Ok(grads)
    }
}
```

**Memory Efficiency:**
- Gradient tensors are allocated during backward pass, not forward
- Intermediate activations can be freed after backward pass if not needed
- Custom ops can control gradient computation memory via `bwd` method

### Error Propagation

**Context Chaining:**
- Error type includes optional backtrace and context
- `.bt()` method adds backtrace to errors
- Custom error types for specific failures (DeviceMismatch, DTypeMismatch, etc.)

```rust
pub enum Error {
    DeviceMismatchBinaryOp { lhs: DeviceLocation, rhs: DeviceLocation, op: &'static str },
    DTypeMismatchBinaryOp { lhs: DType, rhs: DType, op: &'static str },
    Cuda(Box<dyn std::error::Error>),
    Metal(Box<dyn std::error::Error>),
    BackwardNotSupported { op: &'static str },
    // ... more variants
}

impl Error {
    pub fn bt(self) -> Self {  // Add backtrace
        self.with_backtrace()
    }
}
```

### Borrow Checker Implications

**Tensor Cloning:**
- Tensors implement `Clone` via `Arc::clone`, extremely cheap (pointer copy)
- No borrow checker issues when passing tensors across threads
- Storage is shared, so clones reference same underlying data

**Cross-Thread Operations:**
```rust
use std::thread;

// Thread-safe tensor cloning
let tensor_clone = tensor.clone();  // Cheap Arc clone
thread::spawn(move || {
    let result = tensor_clone.matmul(&other_tensor)?;
    // OK: tensor_clone owns its reference
});

// Potential issue: concurrent writes
thread::spawn(move || {
    let mut storage = tensor.storage().write()?;  // Acquires RwLock write lock
    // ... modify storage
});
thread::spawn(move || {
    let _ = tensor.storage().read()?;  // Blocks until write lock released
});
```

**Lifetime Management:**
- Tensors outlive their storage due to Arc reference counting
- No lifetime annotations needed for tensor operations
- Backend storage manages its own GPU/CPU resources

**In-Place Operations:**
- Require exclusive write lock on storage
- Can block other threads accessing same tensor
- Prefer functional style (return new tensor) for thread-safety

## 3. High-Value Implementation Snippets

### Device Initialization and Basic Tensor Operations

```rust
use candle::{Tensor, DType, Device, Result};

// Initialize devices
let cpu_device = Device::Cpu;
#[cfg(feature = "cuda")]
let cuda_device = Device::new_cuda(0)?;  // GPU 0

// Basic tensor creation
let zeros = Tensor::zeros((2, 3, 4), DType::F32, &cuda_device)?;
let ones = Tensor::ones((2, 3), DType::F32, &cpu_device)?;
let random = Tensor::randn(0.0, 1.0, (10, 20), &cuda_device)?;

// From slices (CPU to GPU transfer)
let data = vec![1.0f32; 100];
let tensor_from_slice = Tensor::from_slice(&data, (10, 10), &cuda_device)?;

// Device transfer
let on_gpu = tensor_from_slice.to_device(&cuda_device)?;

// Basic operations (automatically handle broadcasting)
let a = Tensor::arange(0f32, 6f32, &cuda_device)?.reshape((2, 3))?;
let b = Tensor::arange(0f32, 12f32, &cuda_device)?.reshape((3, 4))?;
let c = a.matmul(&b)?;  // Matrix multiplication

// Element-wise operations
let d = c.add(&ones)?;
let e = c.mul(&random)?;
let f = c.relu()?;

// Reduction operations
let sum = c.sum_all()?;
let mean = c.mean(0)?;  // Mean along dimension 0
let max_vals = c.max(0)?;

// Reshape and transpose (zero-copy operations)
let reshaped = c.reshape((3, 2, 4))?;
let transposed = c.t()?;

# Ok::<(), candle::Error>(())
```

### Model Loading from Safetensors

```rust
use candle::{Device, DType, Tensor};
use candle::safetensors::{MmapedSafetensors, Load};
use candle_nn::VarBuilder;

// Memory-mapped loading (minimal memory overhead)
let device = &Device::new_cuda(0)?;
let safetensors = MmapedSafetensors::new("model.safetensors")?;

// Load specific tensors
let weight = safetensors.load("model.weight", device, DType::F16)?;
let bias = safetensors.load("model.bias", device, DType::F32)?;

// Using VarBuilder for hierarchical loading
let vb = VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F16, device)?;
let encoder_weight = vb.get((768, 768), "encoder.weight")?;
let encoder_bias = vb.get(768, "encoder.bias")?;

// Nested path loading (model.encoder.layer.0.weight)
let layer_vb = vb.push_prefix("encoder").push_prefix("layer.0");
let layer_weight = layer_vb.get((3072, 768), "weight")?;

# Ok::<(), candle::Error>(())
```

### GGUF Quantized Model Loading

```rust
use candle::{Device, DType};
use candle::quantized::{gguf_file, QTensor};
use std::collections::HashMap;

// Load GGUF file
let device = &Device::new_cuda(0)?;
let mut reader = std::io::BufReader::new(std::fs::File::open("model.gguf")?);
let content = gguf_file::Content::read(&mut reader)?;

// Access metadata
let metadata = &content.metadata;
let tensor_infos = &content.tensor_infos;

// Load quantized tensors
let mut tensors = HashMap::new();
for (name, info) in tensor_infos {
    let qtensor = info.read(&mut reader, content.tensor_data_offset, device)?;
    tensors.insert(name.clone(), qtensor);
}

// Use quantized tensor in computation
let q_weight = tensors.get("model.weight").unwrap();
let input = Tensor::randn(0.0, 1.0, (1, 768), device)?;
let output = q_weight.matmul(&input.to_dtype(DType::F32)?)?;

# Ok::<(), candle::Error>(())
```

### Custom Module Implementation

```rust
use candle::{Module, Result, Tensor};
use candle_nn::{Linear, LayerNorm, VarBuilder};

// Custom transformer encoder layer
struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn: FeedForward,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(d_model, n_heads, vb.pp("self_attn"))?;
        let norm1 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm1"))?;
        let ffn = FeedForward::new(d_model, d_ff, vb.pp("ffn"))?;
        let norm2 = candle_nn::layer_norm(d_model, 1e-5, vb.pp("norm2"))?;
        
        Ok(Self { self_attn, norm1, ffn, norm2 })
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection
        let attn_out = self.self_attn.forward(x)?;
        let x = (x + attn_out)?;
        let x = self.norm1.forward(&x)?;
        
        // Feed-forward with residual connection
        let ffn_out = self.ffn.forward(&x)?;
        let x = (x + ffn_out)?;
        let x = self.norm2.forward(&x)?;
        
        Ok(x)
    }
}

// Usage
let vb = VarBuilder::from_mmaped_safetensors(&[safetensors], DType::F16, device)?;
let layer = TransformerEncoderLayer::new(768, 12, 3072, vb.pp("encoder.layer.0"))?;
let input = Tensor::randn(0.0, 1.0, (32, 128, 768), device)?;  // batch=32, seq_len=128
let output = layer.forward(&input)?;

# Ok::<(), candle::Error>(())
```

### Custom Operation Implementation

```rust
use candle::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

// Custom RMSNorm operation
struct RMSNorm {
    eps: f32,
}

impl CustomOp1 for RMSNorm {
    fn name(&self) -> &'static str {
        "rms_norm"
    }

    // CPU implementation
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (dim1, dim2) = layout.shape().dims2()?;
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        
        let mut dst = Vec::with_capacity(dim1 * dim2);
        for idx1 in 0..dim1 {
            let row = &src[idx1 * dim2..(idx1 + 1) * dim2];
            let variance = row.iter().map(|x| x * x).sum::<f32>();
            let scale = (variance / dim2 as f32 + self.eps).sqrt().recip();
            dst.extend(row.iter().map(|x| x * scale));
        }
        
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }

    // Optional CUDA implementation
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::LaunchConfig;
        use candle::cuda_backend::WrapErr;
        
        let (d1, d2) = layout.shape().dims2()?;
        let dev = storage.device().clone();
        let slice = storage.as_cuda_slice::<f32>()?;
        let slice = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => slice.slice(o1..o2),
        };
        
        let elem_count = layout.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(elem_count) }?;
        
        // Load custom CUDA kernel
        let func = dev.get_or_load_custom_func(
            "rms_norm_f32", 
            "rms_norm_module",
            candle::include_cuda!("rms_norm.ptx")?
        )?;
        
        let cfg = LaunchConfig {
            grid_dim: (d1 as u32, 1, 1),
            block_dim: (d2 as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let mut builder = func.builder();
        builder.arg(&dst);
        builder.arg(&slice);
        candle::builder_arg!(builder, self.eps, d1 as f32, d2 as f32);
        unsafe { builder.launch(cfg) }.w()?;
        
        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, layout.shape().clone()))
    }

    // Optional backward pass
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Compute gradient w.r.t. input
        // Returns None if gradient not supported
        Ok(None)  // Placeholder for gradient computation
    }
}

// Usage
let tensor = Tensor::randn(0.0, 1.0, (32, 768), &device)?;
let normalized = tensor.apply_op1(RMSNorm { eps: 1e-5 })?;

# Ok::<(), candle::Error>(())
```

## 4. Integration Gotchas (Crucial)

### Async Runtime Integration (Tokio)

**Device Creation Outside Async Context:**
```rust
// WRONG: Create device inside async task
async fn bad_example() -> Result<()> {
    let device = Device::new_cuda(0)?;  // Each task creates new CUDA context
    // ... use device
}

// CORRECT: Create device once, reuse across tasks
let device = Arc::new(Device::new_cuda(0)?);
for i in 0..10 {
    let device = device.clone();
    tokio::spawn(async move {
        let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
        // Safe: tasks share same CUDA context via Arc
    });
}
```

**Blocking GPU Operations in Async Code:**
```rust
// PROBLEM: GPU synchronization blocks async runtime
async fn problematic() -> Result<()> {
    let result = tensor.matmul(&other_tensor)?;  // Enqueues GPU operation
    device.synchronize()?;  // BLOCKS entire Tokio runtime
    // ...
}

// SOLUTION: Offload to thread pool
async fn correct() -> Result<()> {
    let device = device.clone();
    let tensor = tensor.clone();
    let other_tensor = other_tensor.clone();
    
    tokio::task::spawn_blocking(move || {
        let result = tensor.matmul(&other_tensor)?;
        device.synchronize()?;  // Blocks thread, not async runtime
        Ok::<_, candle::Error>(result)
    }).await?
}
```

**Tensor Lifetime Across await Points:**
```rust
// PROBLEM: Tensor references across await
async fn bad() -> Result<()> {
    let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
    async_operation().await;  // tensor might be dropped
    tensor.matmul(&other)?;  // ERROR: tensor might be invalid
}

// SOLUTION: Clone tensors before await
async fn good() -> Result<()> {
    let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
    async_operation().await;
    tensor.matmul(&other)?;  // OK: tensor still valid
}
```

### Godot GDExtension Integration

**GDExtension Threading Model:**
```rust
// PROBLEM: Godot's main thread vs worker threads
#[gdextension]
unsafe fn godot_example() {
    let device = Device::new_cuda(0).unwrap();  // Created on main thread
    
    // WRONG: Pass device to worker thread
    std::thread::spawn(move || {
        let tensor = Tensor::zeros((10, 10), DType::F32, &device).unwrap();
        // May crash: CUDA context not properly shared across threads
    });
    
    // CORRECT: Clone Arc-wrapped device
    let device = Arc::new(Device::new_cuda(0).unwrap());
    let device_clone = device.clone();
    std::thread::spawn(move || {
        let tensor = Tensor::zeros((10, 10), DType::F32, &device_clone).unwrap();
        // Safe: Arc ensures proper reference counting
    });
}
```

**GPU Synchronization in Godot Frame Loop:**
```rust
// PROBLEM: GPU operations not synchronized before frame render
#[gdextension]
unsafe fn frame_update(_delta: f64) {
    let tensor = Tensor::randn(0.0, 1.0, (10, 10), DType::F32, &device).unwrap();
    let result = tensor.matmul(&other).unwrap();
    // Frame might render before GPU computation completes
}

// SOLUTION: Explicit synchronization
#[gdextension]
unsafe fn frame_update(_delta: f64) {
    let tensor = Tensor::randn(0.0, 1.0, (10, 10), DType::F32, &device).unwrap();
    let result = tensor.matmul(&other).unwrap();
    device.synchronize().unwrap();  // Wait for GPU completion
    // Safe to render frame
}
```

**Godot Reference Counting vs Arc:**
```rust
// PROBLEM: Mixing Godot RefCounted and Arc
#[derive(GodotClass)]
struct TensorWrapper {
    tensor: Tensor,  // Contains Arc<RwLock<Storage>>
}

// CORRECT: Store device separately to avoid reference cycles
#[derive(GodotClass)]
struct TensorWrapper {
    tensor: Tensor,
    #[init(default = Device::Cpu)]
    device: Device,  // Avoids Arc cycles
}
```

### GPU Synchronization

**Asynchronous GPU Operations:**
```rust
// PROBLEM: Implicit synchronization
let result = tensor.matmul(&other)?;
let cpu_data = result.to_vec2::<f32>()?;  // Implicit sync - blocks CPU

// SOLUTION: Batch operations, minimize sync
let result1 = tensor1.matmul(&other1)?;
let result2 = tensor2.matmul(&other2)?;
let result3 = tensor3.matmul(&other3)?;
device.synchronize()?;  // One explicit sync for all operations
let data1 = result1.to_vec2::<f32>()?;
let data2 = result2.to_vec2::<f32>()?;
let data3 = result3.to_vec2::<f32>()?;
```

**Cross-Device Operations:**
```rust
// PROBLEM: Implicit device transfer
let cpu_tensor = Tensor::zeros((10, 10), DType::F32, &Device::Cpu)?;
let gpu_tensor = Tensor::zeros((10, 10), DType::F32, &cuda_device)?;
let result = cpu_tensor.matmul(&gpu_tensor)?;  // ERROR: Device mismatch

// SOLUTION: Explicit device transfer
let cpu_on_gpu = cpu_tensor.to_device(&cuda_device)?;
let result = cpu_on_gpu.matmul(&gpu_tensor)?;
```

**GPU Memory Leaks:**
```rust
// PROBLEM: Tensors accumulate in loops
for i in 0..1000 {
    let tensor = Tensor::randn(0.0, 1.0, (1000, 1000), DType::F32, &device)?;
    let result = tensor.matmul(&other)?;
    // tensors not explicitly dropped, GPU memory accumulates
}

// SOLUTION: Explicit scope for cleanup
for i in 0..1000 {
    let result = {
        let tensor = Tensor::randn(0.0, 1.0, (1000, 1000), DType::F32, &device)?;
        tensor.matmul(&other)?
    };  // tensor dropped here, GPU memory freed
}

// ALTERNATIVE: Reuse tensors
let mut buffer = Tensor::zeros((1000, 1000), DType::F32, &device)?;
for i in 0..1000 {
    buffer = Tensor::randn(0.0, 1.0, (1000, 1000), DType::F32, &device)?;
    let result = buffer.matmul(&other)?;
}
```

### VRAM Spikes

**Batch Processing:**
```rust
// PROBLEM: All batches in memory simultaneously
let batch_size = 32;
let num_batches = 100;
let mut all_results = Vec::new();
for batch_idx in 0..num_batches {
    let batch = load_batch(batch_idx)?;
    let result = model.forward(&batch)?;
    all_results.push(result);  // Accumulates in VRAM
}

// SOLUTION: Process sequentially, write to disk/stream
for batch_idx in 0..num_batches {
    let batch = load_batch(batch_idx)?;
    let result = model.forward(&batch)?;
    save_to_disk(&result, batch_idx)?;  // Immediately free VRAM
}
```

**Large Intermediate Activations:**
```rust
// PROBLEM: Backward pass retains all activations
let output = model.forward(&input)?;
let loss = output.compute_loss()?;
loss.backward()?;  // All intermediate tensors retained

// SOLUTION: Gradient checkpointing (manual)
let output = {
    let hidden1 = model.layer1.forward(&input)?;
    let hidden2 = model.layer2.forward(&hidden1)?;  // Drop hidden1 here
    model.layer3.forward(&hidden2)?
};
```

**Quantized Models:**
```rust
// PROBLEM: Dequantization creates full precision tensors
let q_weight = load_quantized_weight()?;
let dequantized = q_weight.dequantize()?;  // VRAM spike
let result = dequantized.matmul(&input)?;

// SOLUTION: Use quantized matmul directly
let result = q_weight.matmul(&input.to_dtype(DType::F32)?)?;  // Operates on quantized data
```

### Thread Pool Integration

**Rayon Integration:**
```rust
// PROBLEM: Each thread creates new CUDA context
use rayon::prelude::*;

(0..10).into_par_iter().for_each(|i| {
    let device = Device::new_cuda(0).unwrap();  // Each thread creates new context
    let tensor = Tensor::zeros((10, 10), DType::F32, &device).unwrap();
});

// SOLUTION: Shared device across threads
let device = Arc::new(Device::new_cuda(0)?);
(0..10).into_par_iter().for_each_with(device, |device, i| {
    let tensor = Tensor::zeros((10, 10), DType::F32, device).unwrap();
});
```

**Tokio Task Pool:**
```rust
// PROBLEM: Too many GPU operations overwhelm scheduler
async fn overload() -> Result<()> {
    let tasks: Vec<_> = (0..1000).map(|i| {
        tokio::spawn(async move {
            let tensor = Tensor::zeros((1000, 1000), DType::F32, &device)?;
            tensor.matmul(&other)
        })
    }).collect();
    
    for task in tasks {
        task.await??;  // 1000 concurrent GPU operations
    }
    Ok(())
}

// SOLUTION: Semaphore to limit concurrency
async fn controlled() -> Result<()> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(4));  // Max 4 concurrent
    let tasks: Vec<_> = (0..1000).map(|i| {
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            let tensor = Tensor::zeros((1000, 1000), DType::F32, &device)?;
            tensor.matmul(&other)
        })
    }).collect();
    
    for task in tasks {
        task.await??;
    }
    Ok(())
}
```

### Memory Alignment and Performance

**Contiguous Memory:**
```rust
// PROBLEM: Non-contiguous tensors reduce performance
let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
let sliced = tensor.narrow(0, 0, 5)?;  // Non-contiguous
let result = sliced.matmul(&other)?;  // Slower due to non-contiguous memory

// SOLUTION: Explicitly contiguous
let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
let sliced = tensor.narrow(0, 0, 5)?.contiguous()?;
let result = sliced.matmul(&other)?;  // Faster
```

**Memory Alignment for GPU:**
```rust
// PROBLEM: Misaligned memory transfers
let data: Vec<f32> = vec![0.0; 100];
let tensor = Tensor::from_slice(&data, (10, 10), &device)?;
// May require padding for GPU alignment

// SOLUTION: Use aligned allocations
use aligned_vec::AVec;
let data: AVec<f32, 256> = AVec::from(vec![0.0; 100]);  // 256-byte aligned
let tensor = Tensor::from_slice(&data, (10, 10), &device)?;
// Faster GPU transfers
```

### Critical Warnings Summary

1. **Never create CUDA context inside async loops** - creates memory leaks and crashes
2. **Always synchronize GPU before accessing CPU data** - otherwise data races
3. **Batch GPU operations, minimize sync points** - maximizes throughput
4. **Explicitly handle cross-device transfers** - implicit transfers are errors
5. **Use Arc<Device> for sharing across threads** - ensures proper reference counting
6. **Drop tensors explicitly in loops** - prevents VRAM accumulation
7. **Use quantized ops directly** - avoid dequantization spikes
8. **Limit concurrent GPU operations** - overwhelm scheduler causes stalls
9. **Prefer contiguous memory** - non-contiguous reduces performance
10. **Monitor VRAM usage** - unexpected spikes indicate memory leaks

**Performance Profiling:**
```rust
// Profile GPU operations
use std::time::Instant;

let start = Instant::now();
let result = tensor.matmul(&other)?;
device.synchronize()?;
let elapsed = start.elapsed();
println!("Matmul took: {:?}", elapsed);

// Profile memory usage
let start_mem = device.get_memory_usage()?;
let result = large_computation()?;
let end_mem = device.get_memory_usage()?;
println!("Memory delta: {} MB", (end_mem - start_mem) / 1024 / 1024);
```
