use rand::prelude::*;
use rand::rng;
use rand::seq::index::sample;

pub struct FillInMiddle<'a> {
    pub fim_rate: f32,
    pub psm_spm_split: f32,
    pub file_separator_token: &'a str,
    pub fim_prefix_token: &'a str,
    pub fim_middle_token: &'a str,
    pub fim_suffix_token: &'a str,
}

impl FillInMiddle<'_> {
    pub fn perform_on_document_text(&mut self, document_text: &str) -> String {
        let mut random = rng();

        document_text
            .split(&self.file_separator_token)
            .map(|file_text| {
                // Decide whether we're applying FIM to this file text
                if &mut random.random::<f32>() < &mut self.fim_rate {
                    // Extract into unicode chars because of multi-byte characters
                    let file_chars: Vec<char> = file_text.chars().collect();

                    // Exclude front and rear character indices we don't want to split at
                    let front_offset = 1;
                    let rear_offset = 1;
                    let range_clip = front_offset + rear_offset + 1;

                    // Boundary condition: text is too short to rearrange
                    if range_clip > file_chars.len() || (file_chars.len() - range_clip) < 2 {
                        file_text.to_string()
                    } else {
                        let mut break_points: Vec<usize> =
                            sample(&mut random, file_chars.len() - range_clip, 2)
                                .into_iter()
                                .map(|index| index + front_offset)
                                .collect();
                        break_points.sort();

                        // Slice out the chars and back to utf-8 strings
                        let prefix = file_chars[..break_points[0]].iter().collect::<String>();
                        let middle = file_chars[break_points[0]..break_points[1]]
                            .iter()
                            .collect::<String>();
                        let suffix = file_chars[break_points[1]..].iter().collect::<String>();

                        if &mut random.random::<f32>() < &mut self.psm_spm_split {
                            // Reorder into Prefix-Suffix-Middle
                            format!(
                                "{}{}{}{}{}{}",
                                self.fim_prefix_token,
                                prefix,
                                self.fim_suffix_token,
                                suffix,
                                self.fim_middle_token,
                                middle
                            )
                        } else {
                            // Reorder into Suffix-Prefix-Middle
                            format!(
                                "{}{}{}{}{}{}",
                                self.fim_suffix_token,
                                suffix,
                                self.fim_prefix_token,
                                prefix,
                                self.fim_middle_token,
                                middle
                            )
                        }
                    }
                } else {
                    file_text.to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(&self.file_separator_token)
    }
}
