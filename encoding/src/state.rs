use crate::*;
use z3::*;

#[derive(Debug, Clone, PartialEq)]
pub struct State {
	pub stack: Vec<Value>,
	pub locals: Vec<Value>,
	pub trapped: bool,
}

pub struct EncodedState<'ctx, 'constants> {
	ctx: &'ctx Context,
	constants: &'constants Constants<'ctx>,
	n_locals: usize,
	stack_pointer: FuncDecl<'ctx>,
	stack_func: FuncDecl<'ctx>,
	stack_type_func: FuncDecl<'ctx>,
	local_func: FuncDecl<'ctx>,
	trapped: FuncDecl<'ctx>,
}

impl<'ctx, 'constants> EncodedState<'ctx, 'constants> {
	pub fn new(
		ctx: &'ctx Context,
		constants: &'constants Constants<'ctx>,
		prefix: String,
		suffix: String,
	) -> Self {
		let domain: Vec<Sort<'ctx>> = constants.bounds.iter().map(Ast::sort).collect();
		let domain: Vec<&Sort<'ctx>> = domain.iter().collect();
		let mut domain_int = domain.clone();
		let int_sort = ctx.int_sort();
		domain_int.push(&int_sort);

		Self {
			ctx,
			constants,
			n_locals: constants.initial_locals.len(),
			stack_pointer: ctx.func_decl(
				ctx.str_sym(&format!("{}stack_pointer{}", prefix, suffix)),
				&domain,
				&ctx.int_sort(),
			),
			stack_func: ctx.func_decl(
				ctx.str_sym(&format!("{}stack{}", prefix, suffix)),
				&domain_int,
				&constants.value_type_config.value_sort(ctx),
			),
			stack_type_func: ctx.func_decl(
				ctx.str_sym(&format!("{}stack_type{}", prefix, suffix)),
				&domain_int,
				&constants.value_type_config.value_type_datatype(ctx).sort,
			),
			local_func: ctx.func_decl(
				ctx.str_sym(&format!("{}local{}", prefix, suffix)),
				&domain_int,
				&constants.value_type_config.value_sort(ctx),
			),
			trapped: ctx.func_decl(
				ctx.str_sym(&format!("{}trapped{}", prefix, suffix)),
				&domain,
				&ctx.bool_sort(),
			),
		}
	}

	// stack_pointer - 1 is top of stack
	pub fn stack_pointer(&self) -> Ast<'ctx> {
		let args: Vec<_> = self.constants.bounds.iter().collect();

		self.stack_pointer.apply(&args)
	}

	pub fn stack(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		let mut args: Vec<_> = self.constants.bounds.iter().collect();
		args.push(index);

		self.stack_func.apply(&args)
	}

	pub fn stack_type(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		let mut args: Vec<_> = self.constants.bounds.iter().collect();
		args.push(index);

		self.stack_type_func.apply(&args)
	}

	pub fn local(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		let mut args: Vec<_> = self.constants.bounds.iter().collect();
		args.push(index);

		self.local_func.apply(&args)
	}

	pub fn trapped(&self) -> Ast<'ctx> {
		let args: Vec<_> = self.constants.bounds.iter().collect();

		self.trapped.apply(&args)
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
