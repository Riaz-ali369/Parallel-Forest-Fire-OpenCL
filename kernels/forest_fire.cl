// Simple LCG random generator
inline uint lcg_random(uint *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

// Returns a float in 0, 1
inline float random01(uint *state) {
    return (float)lcg_random(state) / 4294967296.0f;
}

// Initialises the grid with empty, tree or burning cells
kernel void init_forest(
    global int* grid,
    const int total_cells,
    const float probTree,
    const float probBurning,
    const uint seed
) {
    const int id = get_global_id(0);

    if (id >= total_cells) {
        return;
    }

    uint state = seed + (uint)id * 747796405u;

    const float rand_tree = random01(&state);

    // Empty cell
    if (rand_tree >= probTree) {
        grid[id] = 0;
        return;
    }

    const float rand_burning = random01(&state);

    // Tree or burning tree
    if (rand_burning < probBurning) {
        grid[id] = 2;
    } else {
        grid[id] = 1;
    }
}

// One simulation step: applies fire spread rules
kernel void spread_step(
    global const int* current_grid,
    global int* next_grid,
    const int width,
    const int height,
    const float probImmune,
    const float probLightning,
    const uint seed,
    const int step
) {
    const int id = get_global_id(0);
    const int total_cells = width * height;

    if (id >= total_cells) {
        return;
    }

    const int x = id % width;
    const int y = id / width;

    const int current_state = current_grid[id];

    // Empty stays empty
    if (current_state == 0) {
        next_grid[id] = 0;
        return;
    }

    // Burning becomes empty
    if (current_state == 2) {
        next_grid[id] = 0;
        return;
    }

    int burning_neighbour = 0;

    // Check all 8 neighbours (Moore neighbourhood)
    for (int dy = -1; dy <= 1 && !burning_neighbour; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }

            // Wrap around grid edges (periodic boundary)
            const int nx = (x + dx + width) % width;
            const int ny = (y + dy + height) % height;
            const int neighbour_index = ny * width + nx;

            if (current_grid[neighbour_index] == 2) {
                burning_neighbour = 1;
                break;
            }
        }
    }

    // Per cell random state varies by step
    uint state = seed + (uint)id * 747796405u + (uint)step * 104729u;

    const float rand_immune = random01(&state);

    // Immune tree ignores fire this step
    if (rand_immune < probImmune) {
        next_grid[id] = 1;
        return;
    }

    // Fire spreads from neighbour
    if (burning_neighbour) {
        next_grid[id] = 2;
        return;
    }

    const float rand_lightning = random01(&state);

    // Random lightning ignition
    if (rand_lightning < probLightning) {
        next_grid[id] = 2;
    } else {
        next_grid[id] = 1;
    }
}