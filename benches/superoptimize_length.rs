use std::time::Instant;
use wassup::superoptimize_snippet;
use wassup_encoding::{Instruction, ValueTypeConfig};

fn main() {
	for i64_size in &[Some(64)] {
		for i in 1..20 {
			println!("Starting {:?}: I32Eqz * {}", i64_size, i);
			let mut source = Vec::new();
			source.resize(i, Instruction::I32Eqz);

			let start = Instant::now();
			superoptimize_snippet(
				&source,
				&[],
				ValueTypeConfig {
					i32_size: 4,
					i64_size: *i64_size,
				},
			);

			let elapsed = start.elapsed();
			print!("Completed in ");
			if elapsed.as_secs() > 0 {
				print!("{}s", elapsed.as_secs());
			}
			println!(" {}ms", elapsed.subsec_millis());
		}
	}
}
