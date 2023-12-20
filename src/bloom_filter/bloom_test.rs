#[cfg(test)]
use super::BloomFilter;
// n: number of items in filter. p: false positive rate
// m: number of bits in filter. k: number of hashers
// n = ceil(m / (-k / log(1 - exp(log(p) / k))))
// p = pow(1 - exp(-k / (m / n)), k)
// m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
// k = round((m / n) * log(2));

#[cfg(test)]
pub fn simplified_suggest_size(expected_elements: usize, target_fp_rate: f64) -> usize {
    // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    use std::f64::consts::LN_2;
    let theoretical_optimum = (expected_elements as f64 * target_fp_rate.ln() / (-LN_2 * LN_2))
        .ceil()
        .div_euclid(8.0) as usize;
    let suggested_size = theoretical_optimum.next_power_of_two();

    let min_size: usize = 1 << 20; //1 MiB
    let max_size: usize = usize::MAX / 2; // 9E18 bytes 8exbi-bytes
    suggested_size.clamp(min_size, max_size)
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
fn bloom_suggest_size() {
    // it's hard to derive this exactly since the algorithm is doing closest power of 2
    // instead of exact theoretical optimum

    // Define a struct to hold test case data
    struct TestCase {
        elements: usize,
        fp_rate: f64,
        expected_size: usize,
    }

    // Create a vector of test cases
    let test_cases = vec![
        // test for minimum size
        TestCase {
            elements: 4_000,
            fp_rate: 1E-7,
            expected_size: 1024 * 1024,
        },
        // test for average size
        TestCase {
            elements: 1_000_000,
            fp_rate: 1E-4,
            expected_size: 4_194_304,
        },
        // Add more test cases here as needed
    ];

    for test_case in test_cases {
        let tested_size = BloomFilter::suggest_size_in_bytes(test_case.elements, test_case.fp_rate);
        let simplified_size = simplified_suggest_size(test_case.elements, test_case.fp_rate);

        assert_eq!(
            tested_size, test_case.expected_size,
            "Failed for elements: {}, fp_rate: {}",
            test_case.elements, test_case.fp_rate
        );
        assert_eq!(
            tested_size, simplified_size,
            "Failed for elements: {}, fp_rate: {}",
            test_case.elements, test_case.fp_rate
        );
    }
}
