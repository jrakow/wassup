use either::Either;
use parity_wasm::elements::{CodeSection, FunctionSection, Type, TypeSection, ValueType};
use rayon::prelude::*;
use wassup_encoding::*;
use z3::*;

use std::convert::TryInto;
pub use wassup_encoding::ValueTypeConfig;

pub fn superoptimize_module(
	module: &mut parity_wasm::elements::Module,
	value_type_config: ValueTypeConfig,
	translation_validation_value_type_config: Option<ValueTypeConfig>,
	timeout_ms: Option<usize>,
) {
	// Destructure the module
	// Safe, because the sections should be independent
	let (type_section, code_section, function_section): (
		&TypeSection,
		&mut CodeSection,
		&FunctionSection,
	) = unsafe {
		(
			&*(module.type_section().unwrap() as *const _),
			&mut *(module.code_section_mut().unwrap() as *mut _),
			&*(module.function_section().unwrap() as *const _),
		)
	};

	code_section
		.bodies_mut()
		.par_iter_mut()
		.enumerate()
		.for_each(|(index, body)| {
			let func = function_section.entries()[index];
			let signature = &type_section.types()[func.type_ref() as usize];
			let params = match signature {
				Type::Function(f) => f.params(),
			};
			superoptimize_func_body(
				body,
				params,
				value_type_config,
				translation_validation_value_type_config,
				timeout_ms,
			)
		});
}

pub fn superoptimize_func_body(
	func_body: &mut parity_wasm::elements::FuncBody,
	params: &[ValueType],
	value_type_config: ValueTypeConfig,
	translation_validation_value_type_config: Option<ValueTypeConfig>,
	timeout_ms: Option<usize>,
) {
	let mut function = Function::from_wasm_func_body_params(func_body, params);
	let (instructions, local_types) = (&mut function.instructions, &function.local_types);
	instructions.par_iter_mut().for_each(|snippet| {
		if let Either::Left(vec) = snippet {
			if vec.iter().all(|i| match i {
				Instruction::Const(_) => true,
				_ => false,
			}) {
				return;
			}

			if vec.len() == 1 {
				match vec[0] {
					Instruction::GetLocal(_) => return,
					Instruction::Unreachable => return,
					_ => {}
				}
			}

			let optimized =
				superoptimize_snippet(&vec, &local_types, value_type_config, timeout_ms);

			if let Some(conf) = translation_validation_value_type_config {
				if *vec != optimized {
					match snippets_equivalent(&vec, &optimized, &local_types, conf, timeout_ms) {
						Some(false) | None => return,
						_ => {}
					}
				}
			}
			*vec = optimized;
		}
	});

	*func_body = function.to_wasm_func_body();
}

pub fn superoptimize_snippet(
	source_program: &[Instruction],
	local_types: &[ValueType],
	value_type_config: ValueTypeConfig,
	timeout_ms: Option<usize>,
) -> Vec<Instruction> {
	let mut current_best = source_program.to_vec();

	loop {
		match improve_snippet(&current_best, local_types, value_type_config, timeout_ms) {
			Some(better) => current_best = better,
			None => return current_best,
		}

		if current_best.is_empty() {
			return current_best;
		}
	}
}

fn const_bounds<'ctx>(
	ctx: &'ctx Context,
	program: &[Instruction],
	value_type_config: ValueTypeConfig,
) -> Vec<Option<Ast<'ctx>>> {
	program
		.iter()
		.map(|i| match i {
			Instruction::Const(Value::I32(i)) => {
				let size = value_type_config.i32_size;
				let representable = if size < 32 {
					if *i < 0 {
						(-*i as u32) < 1 << size as u32
					} else {
						(*i as u32) < 1 << size as u32
					}
				} else {
					true
				};

				if !representable {
					Some(ctx.fresh_bitvector_const("const", size.try_into().unwrap()))
				} else {
					None
				}
			}
			Instruction::Const(Value::I64(i)) => {
				if let Some(size) = value_type_config.i64_size {
					let representable = if size < 64 {
						if *i < 0 {
							(-*i as u64) < 1 << size as u64
						} else {
							(*i as u64) < 1 << size as u64
						}
					} else {
						true
					};

					if !representable {
						Some(ctx.fresh_bitvector_const("const", size.try_into().unwrap()))
					} else {
						None
					}
				} else {
					None
				}
			}
			_ => None,
		})
		.collect()
}

pub fn improve_snippet(
	source_program: &[Instruction],
	local_types: &[ValueType],
	value_type_config: ValueTypeConfig,
	timeout_ms: Option<usize>,
) -> Option<Vec<Instruction>> {
	let initial_stack = initial_stack_types(source_program, local_types);

	let config = {
		let mut c = Config::default();
		if let Some(timeout_ms) = timeout_ms {
			c.set_timeout_msec(timeout_ms.try_into().unwrap());
		}
		c
	};

	let ctx = Context::new(&config);
	let solver = Solver::new(&ctx);

	let mut initial_locals = Vec::new();
	let mut initial_locals_bounds = Vec::new();
	for ty in local_types {
		if *ty == ValueType::I32 {
			let sort = ctx.bitvector_sort(value_type_config.i32_size as u32);

			let bound = ctx.fresh_const("initial_local", &sort);
			initial_locals_bounds.push(bound.clone());

			initial_locals.push(value_type_config.i32_wrap_as_i64(&ctx, &bound));
		} else if *ty == ValueType::I64 {
			let sort = ctx.bitvector_sort(value_type_config.i64_size.unwrap() as u32);

			let bound = ctx.fresh_const("initial_local", &sort);
			initial_locals_bounds.push(bound.clone());

			initial_locals.push(bound)
		}
	}

	let const_bounds = const_bounds(&ctx, source_program, value_type_config);

	let bounds: Vec<Ast> = initial_locals_bounds
		.iter()
		.cloned()
		.chain(const_bounds.iter().cloned().filter_map(|x| x))
		.collect();
	let constants = Constants::new(
		&ctx,
		&solver,
		bounds,
		initial_locals,
		local_types.to_vec(),
		&initial_stack,
		value_type_config,
	);
	let target_length = source_program.len() - 1;

	let source_execution = Execution::new_with_const_bounds(
		&constants,
		&solver,
		"source_".to_owned(),
		Either::Left(&source_program),
		const_bounds.clone(),
	);
	let source_state = &source_execution.states[source_program.len()];
	let target_execution = Execution::new_with_const_bounds(
		&constants,
		&solver,
		"target_".to_owned(),
		Either::Right(target_length),
		const_bounds,
	);
	let target_state = &target_execution.states[target_length];

	// assert programs are equivalent
	solver.assert(&equivalent(
		&ctx,
		&constants,
		&source_state,
		&target_state,
		&ctx.from_usize(local_types.len()),
	));

	match solver.check() {
		Some(true) => {
			let model = solver.get_model();
			let better = target_execution.decode_program(&model);
			log::info!("Optimized {:?} to {:?}", &source_program, &better);
			Some(better)
		}
		Some(false) => {
			log::info!("Proved {:?} optimal", &source_program);
			None
		}
		None => {
			log::info!("Optimization of {:?} timed out", &source_program);
			None
		}
	}
}

pub fn snippets_equivalent(
	source_program: &[Instruction],
	target_program: &[Instruction],
	local_types: &[ValueType],
	value_type_config: ValueTypeConfig,
	timeout_ms: Option<usize>,
) -> Option<bool> {
	let initial_stack = initial_stack_types(source_program, local_types);

	let config = {
		let mut c = Config::default();
		if let Some(timeout_ms) = timeout_ms {
			c.set_timeout_msec(timeout_ms.try_into().unwrap());
		}
		c
	};

	let ctx = Context::new(&config);
	let solver = Solver::new(&ctx);

	let mut initial_locals = Vec::new();
	let mut initial_locals_bounds = Vec::new();
	for ty in local_types {
		if *ty == ValueType::I32 {
			let sort = ctx.bitvector_sort(value_type_config.i32_size as u32);

			let bound = ctx.fresh_const("initial_local", &sort);
			initial_locals_bounds.push(bound.clone());

			initial_locals.push(value_type_config.i32_wrap_as_i64(&ctx, &bound));
		} else if *ty == ValueType::I64 {
			let sort = ctx.bitvector_sort(value_type_config.i64_size.unwrap() as u32);

			let bound = ctx.fresh_const("initial_local", &sort);
			initial_locals_bounds.push(bound.clone());

			initial_locals.push(bound)
		}
	}

	let source_const_bounds = const_bounds(&ctx, source_program, value_type_config);
	let target_const_bounds = const_bounds(&ctx, target_program, value_type_config);

	let bounds: Vec<Ast> = initial_locals_bounds
		.iter()
		.cloned()
		.chain(
			source_const_bounds
				.iter()
				.chain(target_const_bounds.iter())
				.cloned()
				.filter_map(|x| x),
		)
		.collect();
	let constants = Constants::new(
		&ctx,
		&solver,
		bounds,
		initial_locals,
		local_types.to_vec(),
		&initial_stack,
		value_type_config,
	);

	let source_execution = Execution::new_with_const_bounds(
		&constants,
		&solver,
		"source_".to_owned(),
		Either::Left(&source_program),
		source_const_bounds.clone(),
	);
	let target_execution = Execution::new_with_const_bounds(
		&constants,
		&solver,
		"target_".to_owned(),
		Either::Left(&target_program),
		target_const_bounds.clone(),
	);
	let source_state = &source_execution.states[source_program.len()];
	let target_state = &target_execution.states[target_program.len()];

	// assert programs are equivalent
	solver.assert(&equivalent(
		&ctx,
		&constants,
		&source_state,
		&target_state,
		&ctx.from_usize(local_types.len()),
	));

	let x = solver.check();
	match x {
		Some(true) => {
			log::info!(
				"Programs {:?} and {:?} proven equivalent",
				&source_program,
				&target_program
			);
		}
		Some(false) => {
			log::info!(
				"Programs {:?} and {:?} proven different",
				&source_program,
				&target_program
			);
		}
		None => {
			log::info!(
				"Equivalence of programs {:?} and {:?} timed out",
				&source_program,
				&target_program
			);
		}
	}
	x
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;
	use Value::*;

	const DEFAULT_VALUE_TYPE_CONFIG: ValueTypeConfig = ValueTypeConfig {
		i32_size: 4,
		i64_size: Some(8),
	};

	const DEFAULT_TIMEOUT: Option<usize> = Some(300_000); // 5 min

	#[test]
	fn superoptimize_nop() {
		let source_program = &[Nop];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn superoptimize_const_nop() {
		let source_program = &[Const(I32(1)), Nop];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I32(1))]);
	}

	#[test]
	fn superoptimize_consts_add() {
		let source_program = &[Const(I32(1)), Const(I32(2)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I32(3))]);
	}

	#[test]
	fn superoptimize_consts_add_64bit() {
		let source_program = &[Const(I64(1)), Const(I64(2)), I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I64(3))]);
	}

	#[test]
	fn superoptimize_add0() {
		let source_program = &[Const(I32(0)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn no_superoptimize_setlocal() {
		let source_program = &[Const(I32(3)), SetLocal(0)];

		// no optimization possible, because locals cannot be changed
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, source_program);
	}

	#[test]
	#[ignore]
	fn superoptimize_unreachable_garbage() {
		let source_program = &[Unreachable, GetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Unreachable]);
	}

	#[test]
	#[ignore]
	fn superoptimize_select_0_1() {
		let source_program = &[
			SetLocal(0),
			Const(I32(0)),
			Const(I32(1)),
			GetLocal(0),
			Select,
		];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![I32Eqz]);
	}

	#[test]
	fn superoptimize_eqz3() {
		let source_program = &[I32Eqz, I32Eqz, I32Eqz];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![I32Eqz]);
	}

	#[test]
	fn superoptimize_trapped() {
		let source_program = &[Const(I32(0)), I32Add, Const(I32(0)), Const(I32(3)), I32DivU];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Unreachable]);
	}

	#[test]
	#[ignore] // works, but takes long
	fn superoptimize_int_extend() {
		let source_program = &[Const(I32(3)), I64ExtendUI32, I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I64(3)), I64Add]);
	}

	#[test]
	fn superoptimize_int_wrap() {
		let source_program = &[Const(I64(3)), I32WrapI64, I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I32(-3)), I32Sub]);
	}

	// Increasing number of initial stack values:
	// ```
	// EQZ EQZ EQZ EQZ EQZ
	// EQ  EQZ EQZ EQZ EQZ
	// EQ  EQ  EQZ EQZ EQZ
	// ...
	// EQ  EQ  EQ  EQ  EQ
	// ```
	// Maybe similar for initial local variables

	#[test]
	#[ignore]
	fn superoptimize_drop() {
		let source_program = &[GetLocal(0), Drop];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	#[ignore]
	fn superoptimize_get_local_eq() {
		let source_program = &[GetLocal(0), GetLocal(0), I32Eq];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I32(1))]);
	}

	#[test]
	fn superoptimize_tee_local_set_local() {
		let source_program = &[TeeLocal(0), SetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![SetLocal(0)]);
	}

	#[test]
	fn superoptimize_set_local_get_local() {
		let source_program = &[SetLocal(0), GetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![TeeLocal(0)]);
	}

	#[test]
	fn superoptimize_get_local_set_local() {
		let source_program = &[GetLocal(0), SetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![]);
	}

	// TODO Example: Optimize out redundant computation with local variable. Overwrite local variable at the end to avoid adding a local.

	// TODO
	// Shift example (check cost function otherwise just equivalence):
	// ```
	// (i32.const 2) (i32.mul)
	// ```
	// to
	// ```
	// (i32.const 1) (i32.shl)
	// ```

	// TODO
	// ```
	// (i32.const 0)
	// (i32.sub)
	// ```
	// to
	// ```
	// unary- / epsilon
	// ```

	#[test]
	fn superoptimize_arithmetic() {
		// 3 + (x - 0)
		let source_program = &[Const(I32(0)), I32Sub, Const(I32(3)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			DEFAULT_VALUE_TYPE_CONFIG,
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![Const(I32(-3)), I32Sub]);
	}

	#[test]
	fn no_superoptimize_large_const() {
		let source_program = &[Const(I32(0x1_0000)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 4,
				i64_size: Some(8),
			},
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, source_program);
	}

	#[test]
	fn no_superoptimize_large_const_64() {
		let source_program = &[Const(I64(0x1_0000)), I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 4,
				i64_size: Some(8),
			},
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, source_program);
	}

	#[test]
	fn superoptimize_double_const_add() {
		let source_program = &[GetLocal(1), Const(I32(32)), I32Add, Const(I32(24)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 4,
				i64_size: Some(8),
			},
			DEFAULT_TIMEOUT,
		);
		assert_eq!(target, vec![GetLocal(1), Const(I32(56)), I32Add]);
	}

	// #[test]
	// fn superoptimize_biguint() {
	// 	// TODO
	// 	// convert_i64(a, b) + convert_i64(c, d)
	// 	let source_program = &[
	// 		// convert_i64(a, b)
	// 		GetLocal(0), I64ExtendUI32, Const(I64(32)), I64Shl, GetLocal(1), I64ExtendUI32, I64Add,
	// 		// convert_i64(c, d)
	// 		GetLocal(2), I64ExtendUI32, Const(I64(32)), I64Shl, GetLocal(3), I64ExtendUI32, I64Add,
	// 		// +
	// 		I64Add
	// 	];
	// 	let target = superoptimize_snippet(
	// 		source_program,
	// 		&[ValueType::I32; 4],
	// 		ValueTypeConfig {
	// 			i32_size: 32,
	// 			i64_size: Some(64),
	// 		},
	// 	);
	// 	// convert_i64(a + c, b + d)
	// 	// TODO not equivalent, because b + d can overflow
	// 	assert_eq!(target, vec![
	// 		// a + c
	// 		GetLocal(0), GetLocal(2), I32Add, I64ExtendUI32, Const(I64(32)), I64Shl,
	// 		// b + d
	// 		GetLocal(1), GetLocal(3), I32Add,
	// 		// if b + d overflowed (b + d < b), add 1 to a + c
	// 		GetLocal(1), I64LtU,
	//
	// 		// FIXME
	// 	]);
	// }

	#[test]
	fn translation_validation() {
		let source_program = &[Const(I32(0)), I32Sub, Const(I32(3)), I32Add];
		let target_program = &[Const(I32(3)), I32Add];

		assert!(snippets_equivalent(
			source_program,
			target_program,
			&[],
			ValueTypeConfig {
				i32_size: 6,
				i64_size: Some(12),
			},
			DEFAULT_TIMEOUT,
		)
		.unwrap());
	}

	#[test]
	fn translation_validation_not_equivalent() {
		let source_program = &[Const(I32(0)), I32Sub, Const(I32(3)), I32Add];
		let target_program = &[Const(I32(2)), I32Add];

		assert!(!snippets_equivalent(
			source_program,
			target_program,
			&[],
			ValueTypeConfig {
				i32_size: 6,
				i64_size: Some(12),
			},
			DEFAULT_TIMEOUT,
		)
		.unwrap());
	}
}
