"""Rich-rendered help for the Mesofield CLI.

:class:`RichGroup` is a drop-in :class:`click.Group` that replaces the plain
"Commands:" listing with a colored ``rich`` tree, so the command groupings
are visible at a glance. Everything else (usage line, description, options)
is left to Click's native formatter.

Subgroups render their own commands as a flat tree; the root group nests one
level deep so ``mesofield --help`` shows the whole map. Degrades gracefully
to Click's default rendering if ``rich`` is unavailable or output is being
captured oddly.
"""

from __future__ import annotations

import io
import shutil
import sys

import click


def _short(cmd: click.Command) -> str:
    return cmd.get_short_help_str(limit=70)


def _add_leaves(node, items) -> None:
    """Add ``(name, command)`` leaves to ``node``, name-padded for alignment."""
    width = max((len(name) for name, _ in items), default=0)
    for name, cmd in items:
        node.add(f"[green]{name.ljust(width)}[/green]   [dim]{_short(cmd)}[/dim]")


class RichGroup(click.Group):
    """Click group that lists its subcommands as a ``rich`` tree.

    Set :attr:`loose_command_heading` on an instance to bucket non-group
    commands under a labeled node (used by the root group to set the
    free-standing acquisition commands apart from the command groups).
    """

    loose_command_heading: str | None = None

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or getattr(cmd, "hidden", False):
                continue
            commands.append((name, cmd))
        if not commands:
            return

        try:
            rendered = self._render_tree(ctx, commands)
        except Exception:
            super().format_commands(ctx, formatter)
            return

        formatter.write("\n")
        formatter.write(rendered)
        formatter.write("\n")

    def _render_tree(self, ctx: click.Context, commands) -> str:
        from rich.console import Console
        from rich.tree import Tree

        groups = [(n, c) for n, c in commands if isinstance(c, click.Group)]
        leaves = [(n, c) for n, c in commands if not isinstance(c, click.Group)]

        tree = Tree(f"[bold]{ctx.command_path}[/bold]", guide_style="dim")

        if leaves:
            if groups and self.loose_command_heading:
                bucket = tree.add(f"[bold blue]{self.loose_command_heading}[/bold blue]")
                _add_leaves(bucket, leaves)
            else:
                _add_leaves(tree, leaves)

        for name, group in groups:
            branch = tree.add(f"[bold cyan]{name}[/bold cyan]   [dim]{_short(group)}[/dim]")
            sub = [
                (sn, sc)
                for sn in group.list_commands(ctx)
                if (sc := group.get_command(ctx, sn)) and not getattr(sc, "hidden", False)
            ]
            if sub:
                _add_leaves(branch, sub)

        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        width = shutil.get_terminal_size((100, 24)).columns
        buf = io.StringIO()
        console = Console(
            file=buf,
            force_terminal=is_tty,
            no_color=not is_tty,
            width=max(40, width),
        )
        console.print(tree)
        return buf.getvalue().rstrip("\n")
