use crate::*;
use z3::*;

#[derive(Debug, Clone, PartialEq)]
pub struct State {
	pub stack: Vec<Value>,
	pub locals: Vec<Value>,
	pub trapped: bool,
}

pub struct EncodedState<'ctx> {
	ctx: &'ctx Context,
	n_locals: usize,
	stack_pointer: Ast<'ctx>,
	stack_func: FuncDecl<'ctx>,
	stack_type_func: FuncDecl<'ctx>,
	local_func: FuncDecl<'ctx>,
	trapped: Ast<'ctx>,
}

impl<'ctx> EncodedState<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		constants: &Constants<'ctx>,
		prefix: String,
		suffix: String,
	) -> Self {
		Self {
			ctx,
			n_locals: constants.initial_locals.len(),
			stack_pointer: ctx.named_int_const(&format!("{}stack_pointer{}", prefix, suffix)),
			stack_func: ctx.func_decl(
				ctx.str_sym(&format!("{}stack{}", prefix, suffix)),
				&[
					&ctx.int_sort(), // stack address
				],
				&constants.value_type_config.value_sort(ctx),
			),
			stack_type_func: ctx.func_decl(
				ctx.str_sym(&format!("{}stack_type{}", prefix, suffix)),
				&[
					&ctx.int_sort(), // index
				],
				&constants.value_type_config.value_type_datatype(ctx).sort,
			),
			local_func: ctx.func_decl(
				ctx.str_sym(&format!("{}local{}", prefix, suffix)),
				&[&ctx.int_sort()],
				&constants.value_type_config.value_sort(ctx),
			),
			trapped: ctx.named_bool_const(&format!("{}trapped{}", prefix, suffix)),
		}
	}

	// stack_pointer - 1 is top of stack
	pub fn stack_pointer(&self) -> Ast<'ctx> {
		self.stack_pointer.clone()
	}

	pub fn stack(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_func.apply(&[&index])
	}

	pub fn stack_type(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_type_func.apply(&[&index])
	}

	pub fn local(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		self.local_func.apply(&[&index])
	}

	pub fn trapped(&self) -> Ast<'ctx> {
		self.trapped.clone()
	}

	pub fn decode(&self, model: &Model<'ctx>, constants: &Constants<'ctx>) -> State {
		let stack_pointer = model
			.eval(&self.stack_pointer())
			.unwrap()
			.as_usize()
			.unwrap();

		State {
			stack: (0..stack_pointer)
				.map(|i| {
					let i = self.ctx.from_usize(i);
					let ty = constants.value_type_config.decode_value_type(
						self.ctx,
						model,
						&self.stack_type(&i),
					);

					Value::decode(&self.stack(&i), model, ty, constants.value_type_config)
				})
				.collect(),
			locals: (0..self.n_locals)
				.map(|i| {
					let i = self.ctx.from_usize(i);

					let ty = constants.value_type_config.decode_value_type(
						self.ctx,
						model,
						&constants.local_type(&i),
					);

					Value::decode(&self.local(&i), model, ty, constants.value_type_config)
				})
				.collect(),
			trapped: model.eval(&self.trapped()).unwrap().as_bool().unwrap(),
		}
	}
}
