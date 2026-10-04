# Chapter 54 Solutions — Key Ideas

- FP16 has more fraction bits than BF16 but far fewer exponent bits and therefore a much smaller dynamic range.
- loss scaling moves tiny FP16 gradients into a representable range, then gradients are unscaled before clipping/update.
- autocast selects operation-specific precision instead of blindly casting every tensor.
- BF16 commonly needs less loss-scaling intervention because its exponent range is FP32-like, though it has lower significand precision.
- total training memory includes much more than parameter storage alone.
