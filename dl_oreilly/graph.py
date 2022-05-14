import pathlib
import subprocess
import tempfile

from .protocol import Function, Variable


class Graphviz:
    def save(self, v: Variable, path: pathlib.Path) -> None:
        lines = self._make_graph(v)
        with tempfile.NamedTemporaryFile(mode="w+t") as f:
            f.writelines(map(lambda x: x + "\n", lines))
            f.flush()
            cmd =["dot", f.name, "-T", "png", "-o", str(path)]
            ret = subprocess.run(cmd)
            # if ret.returncode != 0:
            #     import ipdb; ipdb.set_trace()
            #     pass
            assert ret.returncode == 0

    def _dot_var(self, variable: Variable) -> str:
        label = str(variable).replace("variable(", "")[:-1]
        if len(label) > 100:
            label = label[:100] + "..."
        grad = variable.optional_grad
        if grad is not None:
            label += f"\ngrad={grad.shape}".replace("("," ").replace(")", " ")
        return f'{id(variable)} [label="{label}", color=orange, style=filled]'

    def _dot_func(self, f: Function) -> str:
        label = f.name
        return f'{id(f)} [label="{label}", color=lightblue, style=filled, shape=box]'

    def _dot_connect_from_variable(self, v: Variable) -> list[str]:
        f = v.creator
        if f is None:
            return []
        return [f"{id(f)} -> {id(v)}"]

    def _dot_connect_from_function(self, f: Function) -> list[str]:
        lines: list[str] = []
        for x in f.inputs:
            lines.append(f"{id(x)} -> {id(f)}")
        return lines

    def _make_graph(self, v: Variable) -> list[str]:
        vars_set = set([v])
        funcs_set = set()

        variables = [v]
        while len(variables) > 0:
            v = variables.pop()
            f = v.creator
            if f is None:
                continue

            funcs_set.add(f)
            for x in f.inputs:
                if x not in vars_set:
                    vars_set.add(x)
                    variables.append(x)

        lines: list[str] = ["digraph g {"]
        for v in vars_set:
            lines.append(self._dot_var(v))
            lines.extend(self._dot_connect_from_variable(v))
        for f in funcs_set:
            lines.append(self._dot_func(f))
            lines.extend(self._dot_connect_from_function(f))

        lines.append("}")
        return lines
