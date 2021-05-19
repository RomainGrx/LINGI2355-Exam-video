#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 May 17, 23:18:19
@last modified : 2021 May 19, 23:21:16
"""

from manim import *
import numpy as np


class Title(Scene):
    def construct(self):
        this_is = Tex("This is the presentation of")
        this_is.shift(0 * LEFT + 2 * UP)

        paper = Tex(
            '"Advanced synchronization techniques \
                    for task-based runtime systems"'
        )
        paper.scale(0.75)

        in_5_minutes = Tex("in 5 minutes...")
        in_5_minutes.shift(4 * RIGHT + 3 * DOWN)

        self.play(Write(this_is))

        self.play(FadeIn(paper))

        self.wait()

        self.play(Write(in_5_minutes))

        self.wait()


class CoresEvolution(GraphScene):

    CPU_CORES = {
        2000: 1,
        2001: 2,
        2002: 1,
        2003: 2,
        2004: 2,
        2005: 4,
        2006: 4,
        2007: 4,
        2008: 8,
        2009: 8,
        2010: 12,
        2011: 16,
        2012: 16,
        2013: 24,
        2014: 24,
        2015: 36,
        2016: 36,
        2017: 64,
        2018: 64,
        2019: 128,
        2020: 128,
        2021: 128,
    }

    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            y_axis_label="Number of cores per CPU",
            x_axis_label="Years",
            y_label_position=0.2 * UP + RIGHT,
            y_min=0,
            y_max=129,
            x_min=1999,
            x_max=2021,
            y_labeled_nums=[1, 8, 16, 32, 64, 128],
            x_labeled_nums=np.arange(2000, 2021, 10, dtype=np.uint32),
            graph_origin=3 * DOWN + 4 * LEFT,
            **kwargs,
        )

    def construct(self):

        self.setup_axes(animate=True)

        for x, y in EvolutionCPUGraph.CPU_CORES.items():
            d = Dot(color=BLUE).move_to(self.coords_to_point(x, y))
            self.add(d)
            self.wait()

        self.wait()


class DataDependencies(Scene):
    def construct(self):
        code = """
int A = 42;
// '0x5628030aed40' <- &A

#pragma oss task in(A)
{
    #pragma oss task in(A)
}

#pragma oss task in(A)
{
    #pragma oss task in(A)
}
        """

        rendered_code = Code(
            code=code,
            tab_width=4,
            background="window",
            language="cpp",
            font="Monospace",
        )
        rendered_code.width = 5
        rendered_code.shift(4 * LEFT)

        y_code = rendered_code.get_coord(0)
        x_code = rendered_code.get_coord(1)
        width_code = rendered_code.width
        height_code = rendered_code.height

        self.play(Create(rendered_code))

        self.wait()

        # Arrow decleration
        def step_arrow():
            return rendered_code.line_spacing * DOWN

        def get_initial_arrow(**kwargs):
            dy = 0.0 * rendered_code.height * UP
            arrow = Arrow(start=RIGHT, end=LEFT, **kwargs)
            arrow.next_to(rendered_code, RIGHT)
            arrow.shift(-2 * step_arrow())
            return arrow

        arrow = get_initial_arrow(color=BLUE)

        def get_default_task(name, color, pos):
            circle = Circle(radius=0.5, color=color, fill_opacity=0.5)

            text = Text(name)

            circle_text = VGroup(circle, text)
            circle_text.arrange(IN)

            circle_text.move_to(pos)

            return circle_text

        def get_arrow_between_tasks(task1, task2, title):
            arrow = Arrow(task1, task2, buff=0)
            angle = arrow.get_angle()

            text = Text(title, size=0.4)
            to_shift = text.height

            text.move_to(arrow)

            text.shift(to_shift * UP)
            text.rotate(angle, about_point=arrow.get_center())
            start, end = arrow.get_start_and_end()
            vector = (end - start) / np.sum(end - start)
            text.shift(0.2 * np.array([-vector[0], vector[1], vector[2]]))

            arrow_text = VGroup(arrow, text)

            return arrow_text

        tasks_grid = [(2, 2), (5, 1), (2, -1), (5, -2)]
        tasks_pos = list(map(lambda g: g[0] * RIGHT + g[1] * UP, tasks_grid))

        parent1 = get_default_task("A", BLUE, tasks_pos[0])

        self.play(Create(parent1), Create(arrow))

        self.wait()

        child1 = get_default_task("A", ORANGE, tasks_pos[1])
        parent1_to_child1 = get_arrow_between_tasks(parent1, child1, "Child")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(child1),
            Create(parent1_to_child1),
        )

        self.wait()

        parent2 = get_default_task("A", BLUE, tasks_pos[2])
        parent1_to_parent2 = get_arrow_between_tasks(parent1, parent2, "Successor")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(parent2),
            Create(parent1_to_parent2),
        )

        self.wait()

        child2 = get_default_task("A", ORANGE, tasks_pos[3])
        parent2_to_child2 = get_arrow_between_tasks(parent2, child2, "Child")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(child2),
            Create(parent2_to_child2),
        )

        self.wait()

        all_vec = (
            arrow,
            parent1,
            parent2,
            child1,
            child2,
            parent1_to_child1,
            parent2_to_child2,
            parent1_to_parent2,
        )

        self.play(
            *[Uncreate(v) for v in all_vec], rendered_code.animate.shift(7 * LEFT)
        )
        self.wait()


class ASMStates(Scene):
    def get_data_access_block(self, height=2.5, width=3.7, values=None):
        block = VGroup()
        rect = Rectangle(color=WHITE, height=height, width=width)
        h_line = Line(rect.get_left(), rect.get_right()).shift(0.3 * rect.height * UP)

        title_text = Text("DataAccess A")
        title_text.scale(0.7)
        title_text.set_color(BLUE)
        title_text.move_to(VGroup(h_line, VectorizedPoint(rect.get_top())))

        attrs_text = ["Type:", "Address:", "Successor:", "Child:", "Flags:"]

        attrs = Paragraph(*attrs_text, name="attributes")
        attrs.scale(0.5)
        attrs.next_to(h_line.get_left(), DOWN + RIGHT, buff=SMALL_BUFF)

        values_text = values or ["IN", "0x9F3B", "0x6DFF", "0xF3BB", "0xFF24"]

        values = Paragraph(*values_text, name="values")
        values.scale(0.5)
        values.next_to(h_line.get_right(), DOWN + LEFT, buff=SMALL_BUFF)

        block.add(rect, h_line, title_text, attrs, values)

        return block

    def get_asm(self, width, height, name):
        asm = VGroup()

        rect = RoundedRectangle(color=BLUE, width=width, height=height)
        rect.set_fill(BLUE, opacity=0.05)

        name_asm = Text(name, color=BLUE)
        name_asm.scale(0.4)
        move_to = rect.get_center() + np.array([-rect.height / 2, rect.width / 2, 1])
        name_asm.next_to(rect.get_left() + rect.get_top(), DOWN + RIGHT, MED_SMALL_BUFF)

        asm.add(rect, name_asm)
        return asm

    def get_state(self, bits, radius=0.5, scale_text=0.2):
        circle = Circle(radius=radius, color=LIGHT_BROWN, fill_opacity=0.5)

        text = Text(bits)
        text.move_to(circle)
        text.scale(scale_text)

        circle_text = VGroup(circle, text)

        return circle_text

    def construct(self):
        data_access_block = self.get_data_access_block()
        data_access_block.move_to(ORIGIN)

        self.play(Create(data_access_block))
        self.play(Circumscribe(data_access_block))
        self.wait()

        data_access_block.generate_target()
        data_access_block.target.shift(5.5 * LEFT + 2.5 * UP)
        data_access_block.target.scale(0.75)

        self.play(
            MoveToTarget(data_access_block),
        )
        data_access_block = data_access_block.target

        asm = self.get_asm(
            0.9 * config.frame_width,
            0.45 * config.frame_height,
            "Atomic State Machine",
        )
        asm.shift(1.1 * DOWN)

        surr_flags = Rectangle(
            color=BLUE, width=0.95 * data_access_block.width, height=0.285
        ).move_to(data_access_block.get_center() + 0.58 * DOWN)
        surr_flags.set_fill(BLUE, opacity=0.25)

        self.play(Create(surr_flags))

        self.play(
            Transform(surr_flags, asm),
        )

        arrow_messages = [
            None,
            "Read satisfied",
            "Task finished",
            "Successor registered",
            "Message ack",
        ]
        action_messages = [
            None,
            "Run task",
            None,
            "Send read satisfied\n      to successor",
            "Delete",
        ]

        radius = 0.5
        first_ = asm.get_left() + RIGHT
        last_ = asm.get_right() + LEFT
        dist_states = (last_ - first_) / 4

        states_group = VGroup()
        arrows_group = VGroup()
        actions_group = VGroup()
        for idx, (bits, arrow_message, action) in enumerate(
            zip(range(5), arrow_messages, action_messages)
        ):
            bits = "0b" + "0" * (4 - bits) + "1" * bits

            state = self.get_state(bits, scale_text=0.25, radius=radius).move_to(
                first_ + idx * dist_states
            )

            prev_state = (
                states_group.submobjects[-1] if len(states_group.submobjects) else None
            )

            if prev_state:
                arrow = Arrow(prev_state, state, color=WHITE, buff=0)
                text = Text(arrow_message)
                text.scale(0.3)
                text.next_to(arrow, UP, MED_LARGE_BUFF)
                # text.shift((dist_states - 2 * radius) * LEFT / 5)
                arrow_text_mobj = VGroup(arrow, text)

                self.play(Create(arrow_text_mobj))
                self.wait()

                arrows_group.add(arrow_text_mobj)

            self.play(Create(state))
            self.wait()

            if action:
                # theta = PI / 4
                action_mobj = Paragraph(*action.split("\n"), color=YELLOW)
                action_mobj.scale(0.3)
                action_mobj.next_to(state, DOWN, MED_LARGE_BUFF)
                # action_mobj.rotate(theta, about_point=state.get_center())

                self.play(Write(action_mobj))
                self.wait()
                actions_group.add(action_mobj)

            states_group.add(state)

        self.wait()

        def get_message_mobj(scale=0.25, opacity=1):
            message = SVGMobject("ressources/svg/message.svg")
            message.set_fill(WHITE, opacity=opacity)
            message.scale(scale)
            return message

        def get_lock_mobj(scale=0.25, opacity=1):
            message = SVGMobject("ressources/svg/lock.svg")
            message.set_fill(RED, opacity=opacity)
            message.scale(scale)
            return message

        messages_group = VGroup(*[get_message_mobj() for _ in arrows_group])
        for message, arrow in zip(messages_group, arrows_group):
            message.next_to(arrow, DOWN, SMALL_BUFF)
            message.shift(0.1 * LEFT + OUT)

        self.play(FadeIn(messages_group))
        self.wait()

        locks_group = VGroup()
        for arrow_mobj, message_mobj in zip(arrows_group, messages_group):
            lock_mobj = get_lock_mobj()
            lock_mobj.move_to(message_mobj.get_center())
            locks_group.add(lock_mobj)

        self.play(FadeIn(locks_group), FadeOut(messages_group))

        self.play(
            *[elem.animate.shift(config.frame_width * LEFT) for elem in self.mobjects]
        )

        self.wait()
