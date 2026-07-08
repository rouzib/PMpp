# Distributed FFT

A distributed FFT cannot simply apply a global transform to each x-slab independently. PM++ must move data between real-space slab layout and a spectral layout where the required transform axes are local or arranged for collective communication.

The implementation therefore includes transpose or corner-turn operations around local FFT calls. Custom VJPs define reverse-mode behavior for the forward and inverse distributed transforms, including transposed spectral layouts.

Correctness tests should compare against a reference transform on small meshes before benchmark timing is considered meaningful.
