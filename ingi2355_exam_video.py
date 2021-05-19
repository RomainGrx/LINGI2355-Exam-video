#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 May 17, 23:18:19
@last modified : 2021 May 19, 15:12:10
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

        self.wait(1)

        self.play(Write(in_5_minutes))

        self.wait(3)


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
            self.wait(0.2)

        self.wait(2)


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

        self.wait(0)

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

        self.wait(0)

        child1 = get_default_task("A", ORANGE, tasks_pos[1])
        parent1_to_child1 = get_arrow_between_tasks(parent1, child1, "Child")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(child1),
            Create(parent1_to_child1),
        )

        self.wait(0)

        parent2 = get_default_task("A", BLUE, tasks_pos[2])
        parent1_to_parent2 = get_arrow_between_tasks(parent1, parent2, "Successor")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(parent2),
            Create(parent1_to_parent2),
        )

        self.wait(0)

        child2 = get_default_task("A", ORANGE, tasks_pos[3])
        parent2_to_child2 = get_arrow_between_tasks(parent2, child2, "Child")

        self.play(
            arrow.animate.shift(2 * step_arrow()),
            Create(child2),
            Create(parent2_to_child2),
        )

        self.wait(2)

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
        self.wait(2)
