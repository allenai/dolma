use super::BloomFilter;

impl BloomFilter {
    // Technically, we need 3 out of 4 to calculate the other.
    // But we often want many of these to be either minimum itself or to allow other values to be minima.
    // So we often only need 2 out of 4 ot calculate other two values.
    // n: number of items (expected) in filter.
    // p: (target) false positive rate
    // m: number of bits in filter.
    // k: number of hashers
    // n = ceil(m / (-k / log(1 - exp(log(p) / k))))
    // p = pow(1 - exp(-k / (m / n)), k)
    // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    // k = round((m / n) * log(2));

    pub fn optimal_number_of_hashers(size_in_bytes: usize, expected_elements: usize) -> usize {
        use std::f64::consts::LN_2;
        let n = expected_elements as f64;
        let m = (size_in_bytes * 8) as f64;
        let k = (m / n) * (LN_2);
        k.ceil() as usize
    }

    pub fn prob_of_false_positive(
        size_in_bytes: usize,
        expected_elements: usize,
        num_hashers: usize,
    ) -> f64 {
        let n = expected_elements as f64;
        let m = (size_in_bytes * 8) as f64;
        let k = num_hashers as f64;
        (1.0 - (1.0 - (1.0 / m)).powf(k * n)).powf(k)
    }

    pub fn suggest_size_in_bytes(
        expected_elements: usize,
        desired_false_positive_rate: f64,
    ) -> usize {
        let min_size: usize = 1 << 20; // 1MiB
        let max_size: usize = usize::MAX / 2; // 9E18 bytes 8exbi-bytes
        let mut size_in_bytes = min_size;
        while size_in_bytes < max_size
            && Self::prob_of_false_positive(
                size_in_bytes,
                expected_elements,
                Self::optimal_number_of_hashers(size_in_bytes, expected_elements),
            ) > desired_false_positive_rate
        {
            size_in_bytes *= 2;
        }
        size_in_bytes
    }
}
