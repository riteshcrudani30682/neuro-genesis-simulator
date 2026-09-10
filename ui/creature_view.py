"""Optional Pygame view for multi-agent memory/strategy experiments."""
import colorsys


class CreatureView:
    def __init__(self, config, agents, budget):
        import pygame
        self.pg = pygame
        self.agents = agents
        self.budget = budget
        self.selected = None
        self.size = 18
        self.width = config.width*self.size
        pygame.init()
        self.screen = pygame.display.set_mode((self.width+310, max(500, config.height*self.size)))
        pygame.display.set_caption('Neuro-Genesis V2 | Memory + Strategy + PPO')
        self.font = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()

    def __call__(self, world, result):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise KeyboardInterrupt
            if event.type == pg.MOUSEBUTTONDOWN:
                pos = (event.pos[0]//self.size, event.pos[1]//self.size)
                self.selected = next((i for i, c in world.alive.items() if c.position == pos), self.selected)
        self.screen.fill((17, 23, 32))
        for x, y in world.food:
            pg.draw.circle(self.screen, (110, 222, 133), (x*self.size+9, y*self.size+9), 3)
        for x, y in world.hazards:
            pg.draw.rect(self.screen, (222, 97, 87), (x*self.size+4, y*self.size+4, 10, 10))
        living = world.alive
        if self.selected not in living:
            self.selected = max(living, key=lambda i: living[i].fitness) if living else None
        for i, creature in living.items():
            rgb = colorsys.hsv_to_rgb((creature.generation*.17+.55) % 1, .65, .9)
            x, y = creature.position
            pg.draw.circle(self.screen, tuple(int(v*255) for v in rgb), (x*self.size+9, y*self.size+9), 7)
            if i == self.selected:
                pg.draw.circle(self.screen, (255, 240, 120), (x*self.size+9, y*self.size+9), 9, 2)
        lines = ['NEURO-GENESIS V2', f'Step: {world.step_count}', f'Living creatures: {len(living)}',
                 f'LLM calls: {self.budget.used}/{self.budget.maximum}', '',
                 'Click a creature to inspect', 'Esc: save memory and close', '']
        if self.selected is not None:
            creature = living[self.selected]
            agent = self.agents.get(self.selected)
            lines += [f'Creature ID: {creature.id}', f'Generation: {creature.generation}',
                      f'Energy: {creature.energy:.1f}', f'Fitness: {creature.fitness:.2f}']
            if agent is None:
                lines += ['Newborn: controller starts next tick']
            else:
                lines += [f'Goal: {agent.goal}', f'Memory transitions: {agent.memory.transitions}',
                          f'Memory patterns: {len(agent.memory.patterns)}',
                          f'Controller: {"PPO" if agent.low_level else "heuristic"}']
                if agent.strategy.events:
                    lines += [f'Planner: {agent.strategy.events[-1]["source"]}']
        for index, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (227, 235, 244)), (self.width+12, 14+index*25))
        pg.display.flip()
        self.clock.tick(30)

    def close(self):
        self.pg.quit()
