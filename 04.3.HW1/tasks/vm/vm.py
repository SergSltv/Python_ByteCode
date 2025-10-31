"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator

class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.13/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    BINARY_OPS = {
        0: operator.add,  # +
        1: operator.and_,  # &
        2: operator.floordiv,  # //
        3: operator.lshift,  # <<
        4: operator.matmul,  # @
        5: operator.mul,  # *
        6: operator.mod,  # %
        7: operator.or_,  # |
        8: operator.pow,  # **
        9: operator.rshift,  # >>
        10: operator.sub,  # -
        11: operator.truediv,  # /
        12: operator.xor,  # xor

        13: operator.iadd,  # +=
        14: operator.iand,  # &=
        15: operator.ifloordiv,  # //=
        16: operator.ilshift,  # <<=
        17: operator.imatmul,  # @=
        18: operator.imul,  # *=
        19: operator.imod,  # %=
        20: operator.ior,  # |=
        21: operator.ipow,  # **=
        22: operator.irshift,  # >>=
        23: operator.isub,  # -=
        24: operator.itruediv,  # /=
        25: operator.ixor,  # ^=
    }

    COMPARE_OPS = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        'in': lambda lhs, rhs: operator.contains(rhs, lhs),
        'not in': lambda lhs, rhs: not operator.contains(rhs, lhs),
        'is': operator.is_,
        'is not': operator.is_not,
        'exception match': lambda lhs, rhs: issubclass(type(lhs), rhs) if not isinstance(rhs, tuple) else any(
            issubclass(type(lhs), t) for t in rhs)
    }

    @staticmethod
    def _intrinsic_invalid(_: 'Frame', x: tp.Any) -> tp.Any:
        raise ValueError("Invalid intrinsic operation")

    @staticmethod
    def _intrinsic_print(_: 'Frame', x: tp.Any) -> tp.Any:
        print(x)
        return x

    @staticmethod
    def _intrinsic_import_star(self: 'Frame', module: tp.Any) -> tp.Any:
        if not hasattr(module, '__dict__'):
            raise TypeError("Expected a module for import * intrinsic")
        for name in dir(module):
            if not name.startswith('__'):
                self.globals[name] = getattr(module, name)
        return None

    @staticmethod
    def _intrinsic_stopiteration_error(_: 'Frame', x: tp.Any) -> tp.Any:
        if isinstance(x, StopIteration):
            return x.value
        raise TypeError("Expected StopIteration object")

    @staticmethod
    def _intrinsic_unary_positive(_: 'Frame', x: tp.Any) -> tp.Any:
        return +x

    @staticmethod
    def _intrinsic_list_to_tuple(_: 'Frame', x: tp.Any) -> tp.Any:
        return tuple(x) if isinstance(x, list) else x

    @staticmethod
    def _intrinsic_identity(_: 'Frame', x: tp.Any) -> tp.Any:
        return x

    INTRINSIC_1_OPS = {
        0: _intrinsic_invalid,
        1: _intrinsic_print,
        2: _intrinsic_import_star,
        3: _intrinsic_stopiteration_error,
        4: _intrinsic_identity,
        5: _intrinsic_unary_positive,
        6: _intrinsic_list_to_tuple,
        7: _intrinsic_identity,
        8: _intrinsic_identity,
        9: _intrinsic_identity,
        10: _intrinsic_identity,
        11: _intrinsic_identity,
    }

    @staticmethod
    def _intrinsic2_invalid(_: 'Frame', a: tp.Any, b: tp.Any) -> tp.Any:
        raise ValueError("Invalid intrinsic2 operation")

    @staticmethod
    def _intrinsic2_reraise_star(self: 'Frame', orig: tp.Any, match: tp.Any) -> tp.Any:
        if not isinstance(orig, (ExceptionGroup, BaseExceptionGroup)):
            raise TypeError("orig must be ExceptionGroup or BaseExceptionGroup")
        orig_exceptions = orig.exceptions
        match_exceptions = match.exceptions if isinstance(match, (ExceptionGroup, BaseExceptionGroup)) else [match]
        unmatched = [e for e in orig_exceptions if e not in match_exceptions]
        if not unmatched:
            return None
        use_base = any(not isinstance(e, Exception) for e in unmatched)
        cls = BaseExceptionGroup if use_base else ExceptionGroup
        message = orig.args[0] if orig.args else "unhandled exceptions"
        return cls(message, unmatched)

    @staticmethod
    def _intrinsic2_identity(_: 'Frame', a: tp.Any, b: tp.Any) -> tp.Any:
        return (a, b)

    INTRINSIC_2_OPS = {
        0: _intrinsic2_invalid,
        1: _intrinsic2_reraise_star,
        2: _intrinsic2_identity,
        3: _intrinsic2_identity,
        4: _intrinsic2_identity,
    }


    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self._jump_performed = False
        self.instruction_pointer = 0
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self._last_exception: BaseException | None = None
        self.instructions = list(dis.get_instructions(self.code))

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))

        offset_to_index = {}
        for idx, instr in enumerate(instructions):
            offset_to_index[instr.offset] = idx

        while self.instruction_pointer < len(instructions):
            instruction = instructions[self.instruction_pointer]

            # print(f"Stack contents: {self.data_stack}")
            # print(f"Current IP: {self.instruction_pointer}")
            # print(f"Current op: {instruction.opname}")
            # print(f"Current argval: {instruction.argval}")
            # print(f"{self.globals=}")
            # print(f"{self.locals=}")
            # print()

            opname = instruction.opname.lower() + "_op"

            self._jump_performed = False

            if instruction.arg is not None:
                jump_ops = {
                    'JUMP_FORWARD', 'JUMP_ABSOLUTE', 'POP_JUMP_IF_FALSE',
                    'POP_JUMP_IF_TRUE', 'FOR_ITER', 'JUMP_BACKWARD',
                    'POP_JUMP_IF_NONE', 'POP_JUMP_IF_NOT_NONE',
                    'JUMP_BACKWARD_NO_INTERRUPT', 'JUMP_IF_TRUE_OR_POP',
                    'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_NOT_EXC_MATCH'
                }

                if instruction.opname in jump_ops:
                    target_offset = instruction.argval
                    if instruction.opname == 'JUMP_ABSOLUTE':
                        arg = offset_to_index.get(target_offset, target_offset)
                    else:
                        arg = offset_to_index[target_offset]
                else:
                    arg = instruction.argval

                if opname in {'load_global_op', 'load_attr_op'}:
                    getattr(self, opname)(arg, instruction.arg)
                else:
                    getattr(self, opname)(arg)
            else:
                getattr(self, opname)()

            if not self._jump_performed:
                self.instruction_pointer += 1

        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-CALL
        """
        if len(self.data_stack) < arg + 2:
            raise ValueError("Not enough arguments")

        arguments = self.popn(arg)
        self_or_null = self.pop()
        f = self.pop()
        if self_or_null is None:
            result = f(*arguments)
        else:
            result = f(self_or_null, *arguments)
        self.push(result)

    def call_kw_op(self, argc: int) -> None:
        """
        Description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-CALL_KW
        """
        kw_names = self.pop()
        if not isinstance(kw_names, tuple) or not all(isinstance(name, str) for name in kw_names):
            raise TypeError("Expected a tuple of strings for keyword argument names")

        num_kw_args = len(kw_names)
        if len(self.data_stack) < argc + num_kw_args + 2:
            raise ValueError("Not enough arguments on the stack")

        kw_args = {}
        for name in reversed(kw_names):
            if not self.data_stack:
                raise ValueError("Not enough values for keyword arguments")
            value = self.pop()
            kw_args[name] = value

        pos_args = self.popn(argc) if argc > 0 else []

        self_or_null = self.pop()
        f = self.pop()

        try:
            if self_or_null is None:
                result = f(*pos_args, **kw_args)
            else:
                result = f(self_or_null, *pos_args, **kw_args)
            self.push(result)
        except Exception as e:
            raise type(e)(str(e)) from e

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f'Name {arg} is not defined')

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def load_global_op(self, name: str, namei: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-LOAD_GLOBAL
        """
        null = namei & 1
        if name in self.globals:
            val = self.globals[name]
        elif name in self.builtins:
            val = self.builtins[name]
        else:
            raise NameError(f"name '{name}' is not defined")

        self.push(val)
        if null:
            self.push(None)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()
        self.instruction_pointer = len(self.instructions)

    def return_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = arg
        self.instruction_pointer = len(self.instructions)

    def pop_top_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-POP_TOP
        """
        if len(self.data_stack) == 0:
            return
        self.pop()

    def binary_op_op(self, op: int) -> None:
        rhs = self.pop()
        lhs = self.pop()
        operation = self.BINARY_OPS[op]
        self.push(operation(lhs, rhs))

    def make_function_op(self) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()

        CO_VARARGS = 4
        CO_VARKEYWORDS = 8

        ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
        ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
        ERR_MISSING_POS_ARGS = 'Missing positional arguments'
        ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
        ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

        def bind_args(code, *args: tp.Any, **kwargs: tp.Any) -> dict[str, tp.Any]:
            """Bind values from `args` and `kwargs` to corresponding arguments of `func`

            :param func: function to be inspected
            :param args: positional arguments to be bound
            :param kwargs: keyword arguments to be bound
            :return: `dict[argument_name] = argument_value` if binding was successful,
                     raise TypeError with one of `ERR_*` error descriptions otherwise
            """
            flags = code.co_flags

            arg_count = code.co_argcount
            posonly_count = code.co_posonlyargcount
            kwonly_count = code.co_kwonlyargcount
            varnames = code.co_varnames

            has_varargs = bool(flags & CO_VARARGS)
            has_varkwargs = bool(flags & CO_VARKEYWORDS)

            defaults =  ()
            # defaults = func.__defaults__ or {}
            kwdefaults =  {}
            # kwdefaults = func.__kwdefaults__ or {}

            binding_args = {}

            original_kwargs = kwargs.copy()

            pos_arg_names = varnames[:arg_count]

            required_pos_args_count = arg_count - len(defaults)

            for i in range(posonly_count):
                arg_name = pos_arg_names[i]
                if arg_name in original_kwargs and not has_varkwargs:
                    raise TypeError(ERR_POSONLY_PASSED_AS_KW)

            if len(args) < required_pos_args_count:
                missing_args = []
                for i in range(len(args), required_pos_args_count):
                    arg_name = pos_arg_names[i]
                    if i < posonly_count:
                        missing_args.append(arg_name)
                    else:
                        if arg_name not in kwargs:
                            missing_args.append(arg_name)

                if missing_args:
                    raise TypeError(ERR_MISSING_POS_ARGS)

            max_pos_arg_count = arg_count
            if has_varargs:
                max_pos_arg_count = float('inf')

            if len(args) > max_pos_arg_count and not has_varargs:
                raise TypeError(ERR_TOO_MANY_POS_ARGS)

            for i, arg_name in enumerate(pos_arg_names):
                if i < len(args):
                    binding_args[arg_name] = args[i]
                elif arg_name in kwargs:
                    binding_args[arg_name] = kwargs.pop(arg_name)
                else:
                    binding_args[arg_name] = defaults[i - required_pos_args_count]

            current_index = arg_count
            if has_varargs:
                varargs_values = args[arg_count:] if len(args) > arg_count else ()
                varargs_name = varnames[current_index + kwonly_count]
                binding_args[varargs_name] = varargs_values

            kwnonly_names = varnames[current_index: current_index + kwonly_count]

            for kwarg_name in kwnonly_names:
                if kwarg_name in kwargs:
                    binding_args[kwarg_name] = kwargs.pop(kwarg_name)
                elif kwarg_name in kwdefaults:
                    binding_args[kwarg_name] = kwdefaults[kwarg_name]
                else:
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)

            current_index += kwonly_count

            if has_varkwargs:
                varkwargs_name = varnames[current_index + int(has_varargs)]
                binding_args[varkwargs_name] = kwargs
            elif kwargs:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)

            for i, arg_name in enumerate(pos_arg_names):
                if i < len(args) and arg_name in original_kwargs:
                    if i >= posonly_count:
                        raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                    elif not has_varkwargs:
                        raise TypeError(ERR_POSONLY_PASSED_AS_KW)

            return binding_args

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:

            parsed_args: dict[str, tp.Any] = bind_args(code, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def set_function_attribute_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-SET_FUNCTION_ATTRIBUTE
        """
        func = self.pop()
        value = self.pop()

        if arg == 1:
            func.__defaults__ = value
        elif arg == 2:
            func.__kwdefaults__ = value
        elif arg == 4:
            func.__annotations__ = value
        elif arg == 8:
            func.__closure__ = value
        else:
            raise ValueError(f"Unknown function attribute: {arg}")

        self.push(func)

    def call_function_ex_op(self, flags: int) -> None:
        has_kwargs = (flags & 1) != 0

        if has_kwargs:
            kwargs_dict = self.pop()
        else:
            kwargs_dict = {}

        args_tuple = self.pop()
        self_or_null = self.pop()
        callable_obj = self.pop()

        if not isinstance(args_tuple, (tuple, list)):
            raise TypeError(f"argument after * must be an iterable, not {type(args_tuple).__name__}")

        if has_kwargs and not isinstance(kwargs_dict, dict):
            raise TypeError(f"argument after ** must be a mapping, not {type(kwargs_dict).__name__}")

        if isinstance(args_tuple, list):
            args_tuple = tuple(args_tuple)

        try:
            if self_or_null is None:
                result = callable_obj(*args_tuple, **kwargs_dict)
            else:
                result = callable_obj(self_or_null, *args_tuple, **kwargs_dict)
            self.push(result)
        except Exception as e:
            raise type(e)(f"Error calling {callable_obj} with args={args_tuple}, kwargs={kwargs_dict}: {e}") from e

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.13.7/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def get_iter_op(self) -> None:
        it = self.pop()
        self.push(iter(it))

    def for_iter_op(self, arg: int) -> None:
        iterator = self.top()
        try:
            next_value = next(iterator)
            self.push(next_value)
        except StopIteration:
            self.push(None)
            self.instruction_pointer = arg
            self._jump_performed = True

    def end_for_op(self) -> None:
        self.pop()

    def jump_absolute_op(self, arg: int) -> None:
        self.instruction_pointer = arg
        self._jump_performed = True

    def jump_backward_op(self, arg: int) -> None:
        self.instruction_pointer = arg
        self._jump_performed = True

    def pop_jump_if_false_op(self, arg: int) -> None:
        val = self.pop()
        if not val:
            self.instruction_pointer = arg
            self._jump_performed = True

    def jump_forward_op(self, arg: int) -> None:
        self.instruction_pointer = arg
        self._jump_performed = True

    def pop_jump_if_true_op(self, arg: int) -> None:
        val = self.pop()
        if val:
            self.instruction_pointer = arg
            self._jump_performed = True

    def pop_jump_if_none(self, arg: int) -> None:
        value = self.pop()
        if value is None:
            self.instruction_pointer = arg
            self._jump_performed = True

    def pop_jump_if_not_none(self, arg: int) -> None:
        value = self.pop()
        if value is not None:
            self.instruction_pointer = arg
            self._jump_performed = True

    def jump_backward_no_interrupt_op(self, arg: int) -> None:
        self.instruction_pointer = arg
        self._jump_performed = True

    def jump_if_true_or_pop_op(self, arg: int) -> None:
        value = self.top()
        if value:
            self.instruction_pointer = arg
            self._jump_performed = True
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, arg: int) -> None:
        value = self.top()
        if not value:
            self.instruction_pointer = arg
            self._jump_performed = True
        else:
            self.pop()

    def jump_if_not_exc_match_op(self, arg: int) -> None:
        exc = self.pop()
        expected = self.pop()
        if not isinstance(exc, expected):
            self.instruction_pointer = arg
            self._jump_performed = True

    def unpack_sequence_op(self, arg: int) -> None:
        assert (len(self.top()) == arg)
        elements = self.pop()
        for i in range(arg - 1, -1, -1):
            self.push(elements[i])

    def compare_op_op(self, cmp_str: str) -> None:
        coerce = False
        if cmp_str.endswith(' (as bool)'):
            cmp_str = cmp_str[:-10].strip()
            coerce = True

        if cmp_str not in self.COMPARE_OPS:
            raise NotImplementedError(f"Unsupported comparison: {cmp_str}")

        rhs = self.pop()
        lhs = self.pop()
        result = self.COMPARE_OPS[cmp_str](lhs, rhs)
        if coerce:
            result = bool(result)
        self.push(result)

    def binary_slice_op(self) -> None:
        end = self.pop()
        start = self.pop()
        container = self.pop()
        self.push(container[start:end])

    def binary_subscr_op(self) -> None:
        key = self.pop()
        container = self.pop()
        try:
            result = container.__getitem__(key)
            self.push(result)
        except (IndexError, KeyError, TypeError) as e:
            raise type(e)(str(e)) from e

    def call_intrinsic_1_op(self, operand: int) -> None:
        if not self.data_stack:
            raise ValueError("Stack is empty for CALL_INTRINSIC_1")
        arg = self.pop()
        func = self.INTRINSIC_1_OPS.get(operand)
        if func is None:
            raise ValueError(f"Unknown intrinsic 1 opcode: {operand}")
        result = func(self, arg)
        self.push(result)

    def call_intrinsic_2_op(self, operand: int) -> None:
        if len(self.data_stack) < 2:
            raise ValueError("Not enough arguments for CALL_INTRINSIC_2")
        arg2 = self.pop()
        arg1 = self.pop()
        func = self.INTRINSIC_2_OPS.get(operand)
        if func is None:
            raise ValueError(f"Unknown intrinsic 2 opcode: {operand}")
        result = func(self, arg1, arg2)
        self.push(result)

    def is_op_op(self, invert: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9/library/dis.html#opcode-IS_OP
        """
        right = self.pop()
        left = self.pop()
        if invert:
            self.push(left is not right)
        else:
            self.push(left is right)

    def contains_op_op(self, invert: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9/library/dis.html#opcode-CONTAINS_OP
        """
        item = self.pop()
        container = self.pop()
        if invert:
            self.push(item not in container)
        else:
            self.push(item in container)

    def unary_negative_op(self) -> None:
        value = self.pop()
        self.push(-value)

    def unary_not_op(self) -> None:
        value = self.pop()
        self.push(not value)

    def unary_invert_op(self) -> None:
        value = self.pop()
        self.push(~value)

    def to_bool_op(self) -> None:
        value = self.pop()
        self.push(bool(value))

    def nop_op(self) -> None:
        pass

    def copy_op(self, arg: int) -> None:
        if arg < 0 or arg > len(self.data_stack):
            raise IndexError("Copy arg out of range")
        self.push(self.data_stack[-arg])

    def swap_op(self, arg: int) -> None:
        if len(self.data_stack) < arg:
            return
        top_ind = -1
        other_ind = -arg
        self.data_stack[top_ind], self.data_stack[other_ind] = self.data_stack[other_ind], self.data_stack[top_ind]

    def load_fast_op(self, arg: tp.Any) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])

    def store_fast_op(self, arg: str) -> None:
        value = self.pop()
        self.locals[arg] = value

    def build_list_op(self, count: int) -> None:
        self.push(list(self.popn(count)))

    def list_extend_op(self, arg: int) -> None:
        seq = self.pop()
        list.extend(self.data_stack[-arg], seq)

    def list_append_op(self, arg: int) -> None:
        item = self.pop()
        list.append(self.data_stack[-arg], item)

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def set_add_op(self, arg: int) -> None:
        value = self.pop()
        target_set = self.data_stack[-arg]
        target_set.add(value)

    def set_update_op(self, arg: int) -> None:
        seq = self.pop()
        set.update(self.data_stack[-arg], seq)

    def build_map_op(self, count: int) -> None:
        result = {}
        for _ in range(count):
            value = self.pop()
            key = self.pop()
            result[key] = value
        self.push(result)

    def dict_update_op(self, arg: int) -> None:
        map_ = self.pop()
        dict.update(self.data_stack[-arg], map_)

    def dict_merge_op(self, arg: int) -> None:
        map_dict = self.pop()
        target_dict = self.data_stack[-arg]

        if not isinstance(target_dict, dict) or not isinstance(map_dict, dict):
            raise TypeError("DICT_MERGE expects dictionaries")

        for k in map_dict:
            if k in target_dict:
                raise TypeError(f"duplicate key: {k!r}")

        target_dict.update(map_dict)

    def map_add_op(self, arg: int) -> None:
        value = self.pop()
        key = self.pop()
        dict.__setitem__(self.data_stack[-arg], key, value)

    def build_string_op(self, arg: int) -> None:
        parts = self.popn(arg)
        result = "".join(parts)
        self.push(result)

    def build_slice_op(self, argc: int) -> None:
        if argc == 2:
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end))
        elif argc == 3:
            step = self.pop()
            end = self.pop()
            start = self.pop()
            self.push(slice(start, end, step))

    def store_slice_op(self):
        end = self.pop()
        start = self.pop()
        container = self.pop()
        value = self.pop()
        container[start:end] = value

    def delete_subscr_op(self) -> None:
        key = self.pop()
        container = self.pop()
        del container[key]

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        if not isinstance(keys, tuple):
            raise TypeError("Expected tuple of keys on the stack")

        values = self.popn(count)
        result = {k: v for k, v in zip(keys, values)}
        self.push(result)

    def convert_value_op(self, oparg: str) -> None:
        value = self.pop()
        if oparg is str:
            result = str(value)
        elif oparg is repr:
            result = repr(value)
        elif oparg is ascii:
            result = ascii(value)
        else:
            raise ValueError(f"Unsupported conversion: {oparg}")

        self.push(result)

    def format_simple_op(self) -> None:
        value = self.pop()
        result = value.__format__("")
        self.push(result)

    def store_subscr_op(self) -> None:
        key = self.pop()
        container = self.pop()
        value = self.pop()
        container[key] = value

    def load_attr_op(self, name: str, namei: int) -> None:
        low_bit_set = namei & 1
        obj = self.pop()

        if not low_bit_set:
            value = getattr(obj, name)
            self.push(value)
        else:
            try:
                method = getattr(obj, name)
                if callable(method):
                    self.push(method)
                    self.push(obj)
                else:
                    self.push(None)
                    self.push(method)
            except AttributeError:
                raise AttributeError(f"Object {obj} has no attribute {name}")

    def return_generator_op(self) -> None:
        code = self.code

        def gen():
            frame_copy = Frame(
                code,
                self.builtins,
                self.globals,
                dict(self.locals)
            )
            frame_copy.data_stack = list(self.data_stack)
            yield from frame_copy.run()

        self.return_value = gen()
        self.data_stack.clear()
        self.instruction_pointer = len(list(dis.get_instructions(self.code)))

    def load_assertion_error_op(self) -> None:
        self.push(AssertionError)

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            if not hasattr(self, '_last_exception') or self._last_exception is None:
                raise RuntimeError("No active exception to reraise")
            raise self._last_exception
        elif argc == 1:
            exc = self.pop()
            raise exc
        elif argc == 2:
            cause = self.pop()
            exc = self.pop()
            raise exc from cause
        else:
            raise ValueError(f"Unsupported RAISE_VARARGS argc: {argc}")

    def load_fast_and_clear_op(self, var_name: str) -> None:
        value = self.locals.get(var_name, None)
        self.push(value)
        self.locals[var_name] = None

    def store_fast_store_fast_op(self, arg: int) -> None:
        self.locals[self.code.co_varnames[arg >> 4]] = self.data_stack[-1]
        self.locals[self.code.co_varnames[arg & 15]] = self.data_stack[-2]

    def store_fast_load_fast_op(self, arg: tuple[str, str]) -> None:
        value = self.pop()
        store_var, load_var = arg
        self.locals[store_var] = value
        self.push(self.locals[load_var])

    def load_fast_load_fast_op(self, arg: tuple[str, str]) -> None:
        self.push(self.locals[arg[0]])
        self.push(self.locals[arg[1]])

    def store_attr_op(self, arg: str):
        obj = self.pop()
        value = self.pop()
        setattr(obj, arg, value)

    def delete_attr_op(self, arg: str) -> None:
        obj = self.pop()
        delattr(obj, arg)

    def delete_name_op(self, name: str) -> None:
        if name in self.locals:
            del self.locals[name]
        elif name in self.globals:
            del self.globals[name]
        else:
            raise NameError(f"name '{name}' is not defined =======")

class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
