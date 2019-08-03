use either::Either;
use parity_wasm::elements::{CodeSection, FunctionSection, Type, TypeSection, ValueType};
use rayon::prelude::*;
use wassup_encoding::*;
use z3::*;

pub fn superoptimize_module(module: &mut parity_wasm::elements::Module) {
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
			superoptimize_func_body(body, params)
		});
}

pub fn superoptimize_func_body(
	func_body: &mut parity_wasm::elements::FuncBody,
	params: &[ValueType],
) {
	let mut function = Function::from_wasm_func_body_params(func_body, params);
	for snippet in function.instructions.iter_mut() {
		if let Either::Left(vec) = snippet {
			let optimized = superoptimize_snippet(
				&vec,
				&function.local_types,
				ValueTypeConfig {
					i32_size: 4,
					i64_size: Some(8),
				},
			); // TODO
			*vec = optimized;
		}
	}

	*func_body = function.to_wasm_func_body();
}

pub fn superoptimize_snippet(
	source_program: &[Instruction],
	local_types: &[ValueType],
	value_type_config: ValueTypeConfig,
) -> Vec<Instruction> {
	let mut current_best = source_program.to_vec();

	let initial_stack = initial_stack_types(source_program, local_types);

	loop {
		let config = Config::default();
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

		let constants = Constants::new(
			&ctx,
			&solver,
			initial_locals_bounds,
			initial_locals,
			local_types.to_vec(),
			&initial_stack,
			value_type_config,
		);
		let target_length = current_best.len() - 1;

		let source_execution = Execution::new(
			&constants,
			&solver,
			"source_".to_owned(),
			Either::Left(&current_best),
		);
		let source_state = &source_execution.states[current_best.len()];
		let target_execution = Execution::new(
			&constants,
			&solver,
			"target_".to_owned(),
			Either::Right(target_length),
		);
		let target_state = &target_execution.states[target_length];

		let bounds: Vec<_> = constants.bounds.iter().collect();
		// assert programs are equivalent for all inputs
		solver.assert(&ctx.forall_const(
			&bounds,
			&equivalent(
				&ctx,
				&source_state,
				&target_state,
				&ctx.from_usize(local_types.len()),
			),
		));

		if !solver.check() {
			// already optimal
			return current_best;
		}

		// better version found
		// decode

		let model = solver.get_model();
		current_best = target_execution.decode_program(&model);

		if current_best.is_empty() {
			return current_best;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;
	use Value::*;

	#[test]
	fn superoptimize_nop() {
		let source_program = &[Nop];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn superoptimize_const_nop() {
		let source_program = &[Const(I32(1)), Nop];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(1))]);
	}

	#[test]
	fn superoptimize_consts_add() {
		let source_program = &[Const(I32(1)), Const(I32(2)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(3))]);
	}

	#[test]
	fn superoptimize_consts_add_64bit() {
		let source_program = &[Const(I64(1)), Const(I64(2)), I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I64(3))]);
	}

	#[test]
	fn superoptimize_add0() {
		let source_program = &[Const(I32(0)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	#[ignore]
	fn no_superoptimize_setlocal() {
		let source_program = &[Const(I32(3)), SetLocal(0)];

		// no optimization possible, because locals cannot be changed
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, source_program);
	}

	#[test]
	fn superoptimize_unreachable_garbage() {
		let source_program = &[Unreachable, GetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Unreachable]);
	}

	#[test]
	#[ignore] // TODO
	fn superoptimize_eqz() {
		let source_program = &[I32Eqz, I32Eqz];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![]);

		let source_program = &[I32Eqz, I32Eqz, I32Eqz];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![I32Eqz]);
	}

	#[test]
	fn superoptimize_trapped() {
		let source_program = &[Const(I32(0)), I32Add, Const(I32(0)), Const(I32(3)), I32DivU];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Unreachable]);
	}

	#[test]
	#[ignore]
	fn superoptimize_int_extend() {
		let source_program = &[Const(I32(42)), I64ExtendUI32, I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I64(42)), I64Add]);
	}

	#[test]
	#[ignore]
	fn superoptimize_int_wrap() {
		let source_program = &[Const(I64(42)), I32WrapI64, I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(42)), I32Add]);
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
	fn superoptimize_drop() {
		let source_program = &[GetLocal(0), Drop];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	#[ignore]
	fn superoptimize_locals() {
		let source_program = &[GetLocal(0), GetLocal(0), I32Eq];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(1))]);

		let source_program = &[TeeLocal(0), SetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![SetLocal(0)]);

		let source_program = &[SetLocal(0), GetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![TeeLocal(0)]);

		let source_program = &[GetLocal(0), SetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
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
	#[ignore]
	fn superoptimize_arithmetic() {
		// 3 + (x - 0)
		let source_program = &[Const(I32(0)), I32Sub, Const(I32(3)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(3)), I32Sub]);
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
}
