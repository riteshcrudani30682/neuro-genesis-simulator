"""Passive SVG snapshot renderer; contains no simulation or policy updates."""
import colorsys
from html import escape
from pathlib import Path


def render_svg(world, path, selected_id=None):
    """Render the current world, highlighting a selected or highest-fitness survivor."""
    cfg = world.config
    size = 20
    width, height = cfg.width * size, cfg.height * size
    living = world.alive
    if selected_id is None and living:
        selected_id = max(living, key=lambda i: (living[i].fitness, -i))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height+70}" '
             f'viewBox="0 0 {width} {height+70}">',
             '<rect width="100%" height="100%" fill="#101823"/>']
    for x, y in sorted(world.food):
        parts.append(f'<circle cx="{x*size+10}" cy="{y*size+10}" r="3" fill="#68db87"/>')
    for x, y in sorted(world.hazards):
        parts.append(f'<rect x="{x*size+4}" y="{y*size+4}" width="12" height="12" fill="#e66a61"/>')
    for i, creature in living.items():
        x, y = creature.position
        rgb = colorsys.hsv_to_rgb((creature.generation * .17 + .55) % 1, .55, .9)
        color = '#'+''.join(f'{int(v*255):02x}' for v in rgb)
        stroke = '#ffef8a' if i == selected_id else '#17202c'
        title = escape(f'ID {i}, generation {creature.generation}, energy {creature.energy:.1f}, '
                       f'parent {creature.parent_id}, fitness {creature.fitness:.2f}')
        parts.append(f'<circle cx="{x*size+10}" cy="{y*size+10}" r="8" fill="{color}" '
                     f'stroke="{stroke}" stroke-width="2"><title>{title}</title></circle>')
        parts.append(f'<text x="{x*size+10}" y="{y*size+13}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="7" fill="#101823">{i}</text>')
    text = (f'Step {world.step_count} | Living {len(living)} | Best/selected ID {selected_id}',
            'Green: food | Red: hazard | Creature color: generation | Ring: selected')
    for index, line in enumerate(text):
        parts.append(f'<text x="10" y="{height+25+index*23}" font-family="sans-serif" '
                     f'font-size="13" fill="#e7ecf5">{escape(line)}</text>')
    parts.append('</svg>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts), encoding='utf-8')
