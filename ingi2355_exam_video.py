#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 May 17, 23:18:19
@last modified : 2021 May 20, 13:55:42
"""

from manim import *
import numpy as np
from copy import deepcopy


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
            language="java",
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

        arrow = get_initial_arrow(color=WHITE)

        parent1 = get_default_task("A", BLUE, tasks_pos[0])

        self.play(Create(parent1), Create(arrow))

        self.wait()

        child1 = get_default_task("A", ORANGE, tasks_pos[1])
        parent1_to_child1 = get_arrow_between_tasks(parent1, child1, "Child")

        arrow.generate_target()
        # arrow.target.set_fill(ORANGE, 1)
        arrow.target.shift(2 * step_arrow())

        self.play(
            MoveToTarget(arrow),
            Create(child1),
            Create(parent1_to_child1),
        )

        self.wait()

        parent2 = get_default_task("A", BLUE, tasks_pos[2])
        parent1_to_parent2 = get_arrow_between_tasks(parent1, parent2, "Successor")

        # arrow.target.set_fill(BLUE, 1)
        arrow.target.shift(2 * step_arrow())

        self.play(
            MoveToTarget(arrow),
            Create(parent2),
            Create(parent1_to_parent2),
        )

        self.wait()

        child2 = get_default_task("A", ORANGE, tasks_pos[3])
        parent2_to_child2 = get_arrow_between_tasks(parent2, child2, "Child")

        # arrow.target.set_fill(ORANGE, 1)
        arrow.target.shift(2 * step_arrow())

        self.play(
            MoveToTarget(arrow),
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
        circle = Circle(radius=radius, color=ORANGE, fill_opacity=0.5)

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


class SchedulerWaitFree(Scene):
    def get_cpu_mobj(self, color, scale):
        cpu_mobj = SVGMobject("ressources/svg/cpu.svg")
        if color:
            cpu_mobj.set_fill(color, opacity=1)
        if scale:
            cpu_mobj.scale(scale)
        return cpu_mobj

    def stack(self, mobj, num, title=None):
        stack = VGroup()
        for n in range(num):
            delta = (num - n - 1) * np.array([0.05, 0.05, -1])
            new_mobj = deepcopy(mobj)
            new_mobj.shift(delta)
            stack.add(new_mobj)

        if title:
            title_mobj = Text(title)
            title_mobj.width = 1.25 * stack.width
            title_mobj.next_to(stack, UP, SMALL_BUFF)
            stack.add(title_mobj)

        return stack

    def construct(self):
        # TODO: cpu that talks to say : But how ready threads execute a task?
        _TOP = config.frame_height / 2 * UP
        _BOTTOM = -_TOP
        _RIGHT = config.frame_width / 2 * RIGHT
        _LEFT = -_RIGHT

        N_CPUS = 3
        H_SPACE = config.frame_width / 8
        MARGIN = config.frame_width / 7
        USABLE_WIDTH = config.frame_width - MARGIN
        COL_SPACE = (USABLE_WIDTH) / N_CPUS

        V_LINES = VGroup(
            Line(
                config.top + config.frame_width * LEFT / 2 + (MARGIN) * RIGHT,
                config.bottom + config.frame_width * LEFT / 2 + (MARGIN) * RIGHT,
            )
        )

        for _ in range(N_CPUS):
            new_line = deepcopy(V_LINES.submobjects[-1])
            V_LINES.add(new_line.shift(COL_SPACE * RIGHT))

        def get_center_col(idx):
            return VGroup(
                V_LINES.submobjects[idx], V_LINES.submobjects[idx + 1]
            ).get_center()

        def get_center_margin():
            return VGroup(
                Line(_TOP + _LEFT, _BOTTOM + _LEFT), V_LINES.submobjects[0]
            ).get_center()

        cpu_mobj = self.get_cpu_mobj(None, 0.4)
        queue_mobj = Rectangle(width=0.33 * COL_SPACE, height=0.2 * config.frame_height)
        queue_mobj.set_fill(BLUE, opacity=0.25)

        creators_group = VGroup()
        queues_group = VGroup()
        arrows_group = VGroup()
        for cpu_idx in range(N_CPUS):
            curr_cpu_mobj = deepcopy(cpu_mobj)
            curr_cpu_mobj.move_to(get_center_col(cpu_idx))
            curr_cpu_mobj.shift(_TOP + (MED_SMALL_BUFF + curr_cpu_mobj.height) * DOWN)
            creators_group.add(curr_cpu_mobj)

            curr_queue_mobj = deepcopy(queue_mobj)
            curr_queue_mobj.next_to(curr_cpu_mobj, DOWN, MED_LARGE_BUFF)
            queues_group.add(curr_queue_mobj)

            arrow = Arrow(
                curr_cpu_mobj.get_center() + curr_cpu_mobj.height * DOWN / 2,
                curr_queue_mobj.get_center() + curr_queue_mobj.height * UP / 2,
                buff=0,
            )
            arrows_group.add(arrow)

        text_scale = 0.4
        text_color = WHITE
        margin_texts_group = VGroup()
        creator_mobj = (
            (Text("Creator\nThreads").scale(text_scale).set_fill(text_color, opacity=1))
            .move_to(get_center_margin())
            .shift(creators_group.get_center() * UP)
        )
        margin_texts_group.add(creator_mobj)

        queue_mobj = (
            (Text("Thread\nQueues").scale(text_scale).set_fill(text_color, opacity=1))
            .move_to(get_center_margin())
            .shift(queues_group.get_center() * UP)
        )
        margin_texts_group.add(queue_mobj)

        self.play(
            Create(creators_group),
            Create(queues_group),
            Create(arrows_group),
            Write(creator_mobj),
            Write(queue_mobj),
        )
        self.wait()

        scheduler_queue_mobj = Rectangle(
            width=0.8 * N_CPUS * COL_SPACE, height=0.2 * config.frame_height
        ).next_to(queues_group, DOWN, MED_LARGE_BUFF)
        scheduler_queue_mobj.set_fill(ORANGE, 0.25)

        queues_to_scheduler_arrows_group = VGroup()
        for queue in queues_group.submobjects:
            start = queue.get_center() + queue.height * DOWN / 2
            end = start + (scheduler_queue_mobj.get_top() - queues_group.get_bottom())
            arrow = Arrow(start, end, buff=0)
            queues_to_scheduler_arrows_group.add(arrow)

        scheduler_queue_text_mobj = (
            (
                Text("Scheduler\n   Queue")
                .scale(text_scale)
                .set_fill(text_color, opacity=1)
            )
            .move_to(get_center_margin())
            .shift(scheduler_queue_mobj.get_center() * UP)
        )
        margin_texts_group.add(scheduler_queue_text_mobj)

        self.play(
            Create(scheduler_queue_mobj),
            Create(queues_to_scheduler_arrows_group),
            Write(scheduler_queue_text_mobj),
        )
        self.wait()

        workers_group = VGroup()
        scheduler_to_workers_arrows_group = VGroup()
        for cpu_idx in range(N_CPUS):
            curr_cpu_mobj = deepcopy(cpu_mobj)
            curr_cpu_mobj.move_to(get_center_col(cpu_idx))
            curr_cpu_mobj.shift(
                (scheduler_queue_mobj.height + MED_SMALL_BUFF + curr_cpu_mobj.height)
                * DOWN
            )
            workers_group.add(curr_cpu_mobj)

            arrow = Arrow(
                curr_cpu_mobj.get_center() * RIGHT
                + scheduler_queue_mobj.get_bottom() * UP,
                curr_cpu_mobj.get_center() * RIGHT + curr_cpu_mobj.get_top() * UP,
                buff=0,
            )
            scheduler_to_workers_arrows_group.add(arrow)

        worker_mobj = (
            (Text(" Worker\nThreads").scale(text_scale).set_fill(text_color, opacity=1))
            .move_to(get_center_margin())
            .shift(workers_group.get_center() * UP)
        )
        margin_texts_group.add(worker_mobj)

        self.play(
            Create(workers_group),
            Create(scheduler_to_workers_arrows_group),
            Write(worker_mobj),
        )
        self.wait()

        lock_creator_queues_mobj = VGroup()
        for cpu, queue in zip(creators_group, queues_group):
            height = (
                SMALL_BUFF
                + (cpu.get_top() - queue.get_center() - 0.15 * queue.height)[1]
            )
            lock_fill_mobj = RoundedRectangle(
                width=1.25 * queue.width, height=height, color=RED
            )
            lock_fill_mobj.set_fill(RED, opacity=0.1)
            lock_fill_mobj.move_to(
                cpu.get_center() * RIGHT
                + (cpu.get_top() + SMALL_BUFF - lock_fill_mobj.height / 2) * UP
            )

            lock_mobj = get_lock_mobj(scale=0.15)
            lock_mobj.next_to(
                queue.get_right() * RIGHT + queue.get_top() * UP,
                LEFT + DOWN,
                SMALL_BUFF,
            )

            producer_mobj = Text("Producer").scale(text_scale).set_fill(RED, 1)
            producer_mobj.rotate(PI / 2)
            producer_mobj.next_to(lock_fill_mobj, LEFT, SMALL_BUFF)

            lock_creator_queues_mobj.add(
                VGroup(lock_fill_mobj, lock_mobj, producer_mobj)
            )

        height = (
            SMALL_BUFF
            + (
                queue.get_center()
                - scheduler_queue_mobj.get_bottom()
                - 0.15 * queue.height
            )[1]
        )
        lock_fill_scheduler_mobj = RoundedRectangle(
            width=1.1 * scheduler_queue_mobj.width, height=height, color=RED
        )
        lock_fill_scheduler_mobj.set_fill(RED, 0.1)
        lock_fill_scheduler_mobj.move_to(
            scheduler_queue_mobj.get_center()
            + (
                SMALL_BUFF
                + scheduler_queue_mobj.height / 2
                - lock_fill_scheduler_mobj.height / 2
            )
            * DOWN
        )

        consummer_mobj = (
            Text("Consumer")
            .scale(text_scale)
            .set_fill(RED, 1)
            .rotate(PI / 2)
            .next_to(lock_fill_scheduler_mobj, LEFT, SMALL_BUFF)
        )
        print(lock_fill_scheduler_mobj.get_left())
        print(lock_fill_scheduler_mobj.get_top())

        lock_mobj = get_lock_mobj(scale=0.15)
        lock_mobj.next_to(
            lock_fill_scheduler_mobj.get_left() * RIGHT
            + lock_fill_scheduler_mobj.get_top() * UP,
            DOWN + RIGHT,
            MED_SMALL_BUFF,
        )

        lock_scheduler_mobj = VGroup(
            lock_fill_scheduler_mobj, consummer_mobj, lock_mobj
        )

        self.play(Create(lock_creator_queues_mobj), Create(lock_scheduler_mobj))
        self.wait()
