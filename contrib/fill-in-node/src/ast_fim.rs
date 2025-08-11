use rand::prelude::*;
use rand::rng;
use tree_sitter::{Language, Parser as TreeParser, Node};

pub struct AstFillInMiddle<'a> {
    pub fim_rate: f32,
    pub psm_spm_split: f32,
    pub file_separator_token: &'a str,
    pub fim_prefix_token: &'a str,
    pub fim_middle_token: &'a str,
    pub fim_suffix_token: &'a str,
}

impl<'a> AstFillInMiddle<'a> {
    pub fn new(
        fim_rate: f32,
        psm_spm_split: f32,
        file_separator_token: &'a str,
        fim_prefix_token: &'a str,
        fim_middle_token: &'a str,
        fim_suffix_token: &'a str,
    ) -> Self {
        Self {
            fim_rate,
            psm_spm_split,
            file_separator_token,
            fim_prefix_token,
            fim_middle_token,
            fim_suffix_token,
        }
    }

    pub fn perform_on_document_text(&mut self, document_text: &str) -> String {
        let mut random = rng();

        document_text
            .split(self.file_separator_token)
            .map(|file_text| {
                // Decide whether we're applying FIM to this file text
                if random.random::<f32>() < self.fim_rate {
                    // Try to detect language and apply AST-based FIM
                    if let Some(language) = detect_language_from_content(file_text) {
                        match self.apply_ast_fim(file_text, &language, &mut random) {
                            Ok(transformed) => transformed,
                            Err(_) => {
                                // Fallback to character-level FIM if AST parsing fails
                                self.apply_character_fim(file_text, &mut random)
                            }
                        }
                    } else {
                        // Fallback to character-level FIM for unsupported languages
                        self.apply_character_fim(file_text, &mut random)
                    }
                } else {
                    file_text.to_string()
                }
            })
            .collect::<Vec<String>>()
            .join(self.file_separator_token)
    }

    fn apply_ast_fim(&mut self, file_text: &str, language: &str, random: &mut ThreadRng) -> Result<String, Box<dyn std::error::Error>> {
        let lang = get_language_for_name(language)?;
        let mut parser = TreeParser::new();
        parser.set_language(&lang)?;
        
        let tree = parser.parse(file_text, None).ok_or("Failed to parse")?;
        let root_node = tree.root_node();
        
        // Collect all suitable nodes for FIM
        let suitable_nodes = collect_suitable_nodes(root_node, file_text.as_bytes());
        
        if suitable_nodes.len() < 2 {
            // Not enough nodes for meaningful FIM, fall back to character-level
            return Ok(self.apply_character_fim(file_text, random));
        }
        
        // Select a random node as the "middle" part  
        let middle_idx = random.random_range(1..suitable_nodes.len()-1);
        let middle_node = suitable_nodes[middle_idx];
        
        // Split the text based on the selected node
        let middle_start = middle_node.start_byte();
        let middle_end = middle_node.end_byte();
        
        let prefix = &file_text[..middle_start];
        let middle = &file_text[middle_start..middle_end];
        let suffix = &file_text[middle_end..];
        
        // Format according to PSM/SPM split
        if random.random::<f32>() < self.psm_spm_split {
            // Prefix-Suffix-Middle
            Ok(format!(
                "{}{}{}{}{}{}",
                self.fim_prefix_token,
                prefix,
                self.fim_suffix_token,
                suffix,
                self.fim_middle_token,
                middle
            ))
        } else {
            // Suffix-Prefix-Middle
            Ok(format!(
                "{}{}{}{}{}{}",
                self.fim_suffix_token,
                suffix,
                self.fim_prefix_token,
                prefix,
                self.fim_middle_token,
                middle
            ))
        }
    }

    // Fallback character-level FIM (similar to original implementation)
    fn apply_character_fim(&mut self, file_text: &str, random: &mut ThreadRng) -> String {
        let file_chars: Vec<char> = file_text.chars().collect();

        // Exclude front and rear character indices we don't want to split at
        let front_offset = 1;
        let rear_offset = 1;
        let range_clip = front_offset + rear_offset + 1;

        // Boundary condition: text is too short to rearrange
        if range_clip > file_chars.len() || (file_chars.len() - range_clip) < 2 {
            return file_text.to_string();
        }

        let mut break_points: Vec<usize> = (0..2)
            .map(|_| random.random_range(front_offset..file_chars.len() - rear_offset))
            .collect();
        break_points.sort();
        break_points.dedup();

        // Ensure we have exactly 2 distinct break points
        while break_points.len() < 2 {
            let new_point = random.random_range(front_offset..file_chars.len() - rear_offset);
            if !break_points.contains(&new_point) {
                break_points.push(new_point);
            }
        }
        break_points.sort();

        // Slice out the chars and back to utf-8 strings
        let prefix = file_chars[..break_points[0]].iter().collect::<String>();
        let middle = file_chars[break_points[0]..break_points[1]]
            .iter()
            .collect::<String>();
        let suffix = file_chars[break_points[1]..].iter().collect::<String>();

        if random.random::<f32>() < self.psm_spm_split {
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
}

fn detect_language_from_content(content: &str) -> Option<String> {
    // Simple heuristics to detect language from content
    if content.contains("def ") || content.contains("import ") || content.contains("class ") {
        Some("python".to_string())
    } else if content.contains("fn ") || content.contains("use ") || content.contains("struct ") {
        Some("rust".to_string())
    } else if content.contains("function ") || content.contains("const ") || content.contains("let ") {
        if content.contains("interface ") || content.contains(": string") {
            Some("typescript".to_string())
        } else {
            Some("javascript".to_string())
        }
    } else if content.contains("public class ") || content.contains("import java") {
        Some("java".to_string())
    } else if content.contains("#include") || content.contains("int main") {
        Some("cpp".to_string())
    } else if content.contains("package ") || content.contains("func ") {
        Some("go".to_string())
    } else if content.contains("using ") || content.contains("namespace ") {
        Some("csharp".to_string())
    } else {
        None
    }
}

fn get_language_for_name(lang_name: &str) -> Result<Language, Box<dyn std::error::Error>> {
    match lang_name.to_lowercase().as_str() {
        "python" => Ok(tree_sitter_python::language()),
        "rust" => Ok(tree_sitter_rust::language()),
        "cpp" | "c++" => Ok(tree_sitter_cpp::language()),
        "typescript" => Ok(tree_sitter_typescript::language_typescript()),
        "javascript" => Ok(tree_sitter_javascript::language()),
        "java" => Ok(tree_sitter_java::language()),
        "go" => Ok(tree_sitter_go::language()),
        "csharp" | "c#" => Ok(tree_sitter_c_sharp::language()),
        _ => Err(format!("Unsupported language: {}", lang_name).into()),
    }
}

fn collect_suitable_nodes<'a>(node: Node<'a>, source: &'a [u8]) -> Vec<Node<'a>> {
    let mut nodes = Vec::new();
    collect_nodes_recursive(node, source, &mut nodes);
    nodes
}

fn collect_nodes_recursive<'a>(node: Node<'a>, source: &'a [u8], nodes: &mut Vec<Node<'a>>) {
    // Only consider nodes that represent complete constructs
    if is_suitable_for_fim(node, source) {
        nodes.push(node);
    }
    
    // Recursively collect from children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_nodes_recursive(child, source, nodes);
        }
    }
}

fn is_suitable_for_fim(node: Node, _source: &[u8]) -> bool {
    let kind = node.kind();
    let byte_range = node.end_byte() - node.start_byte();
    
    // Skip very small nodes (less than 10 bytes)
    if byte_range < 10 {
        return false;
    }
    
    // Skip comment and string nodes
    if kind.contains("comment") || kind.contains("string") {
        return false;
    }
    
    // Include common statement/expression types that make good FIM candidates
    match kind {
        // Function-like constructs
        "function_definition" | "function_declaration" | "function_item" | "method_declaration" | "method_definition" => true,
        
        // Class/struct/type definitions
        "class_definition" | "class_declaration" | "class_specifier" | "struct_item" | "impl_item" | "type_declaration" => true,
        
        // Control flow statements
        "if_statement" | "if_expression" => true,
        "for_statement" | "for_expression" => true,
        "while_statement" | "while_expression" => true,
        "match_expression" => true,
        
        // Block constructs
        "block" | "compound_statement" => true,
        
        // Other statement types
        "with_statement" | "try_statement" | "expression_statement" => true,
        
        // General statement/expression/declaration pattern (catch remaining valid types)
        _ if kind.ends_with("_statement") || kind.ends_with("_expression") || kind.ends_with("_declaration") => true,
        
        _ => false,
    }
}