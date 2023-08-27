#[cfg(test)]
mod tests {
    use super::super::BloomFilter;

    // n: number of items in filter. p: false positive rate
    // m: number of bits in filter. k: number of hashers
    // n = ceil(m / (-k / log(1 - exp(log(p) / k))))
    // p = pow(1 - exp(-k / (m / n)), k)
    // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    // k = round((m / n) * log(2));

    #[test]
    fn bloom_test_prob_of_false_positive() {
        // calculated from https://hur.st/bloomfilter/
        let size_in_bytes = 1_000_000_000;
        let expected_elements = 1_000_000_000;
        let num_hashers = 8;
        assert_eq!(
            BloomFilter::prob_of_false_positive(size_in_bytes, expected_elements, num_hashers),
            0.025_491_740_593_406_025 as f64
        );
        assert_eq!(
            BloomFilter::prob_of_false_positive(1_048_576, 524288, 2),
            0.013_806_979_447_406_826 as f64
        )
    }
    #[test]
    fn bloom_optimal_hasher_number() {
        let size_in_bytes = 1_000_000_000;
        let expected_elements = 1_000_000_000;
        assert_eq!(
            BloomFilter::optimal_number_of_hashers(size_in_bytes, expected_elements),
            6
        );
        assert_eq!(
            BloomFilter::optimal_number_of_hashers(1_000_000, 500_000),
            12
        )
    }
}
