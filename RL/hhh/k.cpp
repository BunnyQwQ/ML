#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

constexpr int WIDTH  = 7;
constexpr int HEIGHT = 6;
constexpr int H1     = HEIGHT + 1;
constexpr int CELLS  = WIDTH * HEIGHT;


struct Board {
    uint64_t pos[2] = {0, 0};
    std::array<int, WIDTH> heights{};
    int player = 0;
    int moves  = 0;
};

inline bool wins(uint64_t p) {
    uint64_t m;

    m = p & (p >> 7);
    if (m & (m >> 14)) return true;

    m = p & (p >> 1);
    if (m & (m >> 2)) return true;

    m = p & (p >> 8);
    if (m & (m >> 16)) return true;

    m = p & (p >> 6);
    if (m & (m >> 12)) return true;

    return false;
}

inline bool can_play(const Board& b, int col) {
    return b.heights[col] < HEIGHT;
}

inline void play(Board& b, int col) {
    uint8_t ind = col * H1 + b.heights[col];
    b.pos[b.player] |= ind >> 1;
    b.heights[col]++;
    b.moves++;
    b.player ^= 1;
}

inline bool is_terminal(const Board& b) {
    return wins(b.pos[1 - b.player]) || b.moves == CELLS;
}

struct Node {
    int parent = -1;
    std::array<int, WIDTH> children;
    std::vector<int> untried;
    int    N    = 0;
    double W    = 0.;
    int    move = -1;

    Node() { children.fill(-1); }
};

static int make_node(std::vector<Node>& pool, int parent, int move, const Board& b) {
    Node n;
    n.parent = parent;
    n.move   = move;
    if (!is_terminal(b)) {
        for (int c = 0; c < WIDTH; ++c)
            if (can_play(b, c)) n.untried.push_back(c);
    }
    pool.push_back(std::move(n));
    return static_cast<int>(pool.size()) - 1;
}

static int select_child(const std::vector<Node>& pool, int node, double c_puct) {
    const Node& parent = pool[node];
    double log_np = std::log(static_cast<double>(parent.N));

    int    best_idx   = -1;
    double best_score = -1e18;

    for (int col = 0; col < WIDTH; ++col) {
        int ch = parent.children[col];
        if (ch == -1) continue;

        double q = pool[ch].W / pool[ch].N;

        double score = -q + c_puct * sqrt(log_np - std::log(static_cast<double>(pool[ch].N)));

        if (score > best_score) {
            best_score = score;
            best_idx   = ch;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// ЗАДАНИЕ 4: случайная доигровка
//
// Возвращает результат С ТОЧКИ ЗРЕНИЯ ИГРОКА, КОТОРЫЙ ХОДИТ В b НА ВХОДЕ:
//   +1 победа, -1 поражение, 0 ничья.
//
// Цикл:
//   1) если предыдущий игрок (1 - b.player) собрал четвёрку -> партия окончена
//      (тогда тот, чья очередь ходить, ПРОИГРАЛ)
//   2) если b.moves == CELLS -> ничья
//   3) иначе выбрать случайный легальный столбец и сыграть
//
// Board передаётся ПО ЗНАЧЕНИЮ — портить дерево нельзя.
// ---------------------------------------------------------------------------
static double simulate(Board b, std::mt19937& rng) {
    int me = b.player;   // от чьего лица считаем результат

    while (true) {
        if
    }
}

static void backup(std::vector<Node>& pool, int node, double value) {
    while (node != -1) {
        pool[node].N++;
        pool[node].W += value;
        value = -value;
        node = pool[node].parent;
    }
}

static int search(const Board& root_board, int iterations, std::mt19937& rng,
                  double c_puct = 1.41) {
    std::vector<Node> pool;
    pool.reserve(iterations + 1);
    make_node(pool, -1, -1, root_board);

    for (int it = 0; it < iterations; ++it) {
        Board b = root_board;
        int node = 0;

        // 1. SELECTION — спуск, пока узел полностью развёрнут и не терминален
        while (pool[node].untried.empty() && !is_terminal(b)) {
            node = select_child(pool, node, c_puct);
            play(b, pool[node].move);
        }

        // 2. EXPANSION — добавляем ОДИН новый узел
        if (!pool[node].untried.empty()) {
            auto& untried = pool[node].untried;
            std::uniform_int_distribution<int> pick(0, static_cast<int>(untried.size()) - 1);
            int idx = pick(rng);
            int col = untried[idx];
            untried[idx] = untried.back();
            untried.pop_back();

            play(b, col);
            int child = make_node(pool, node, col, b);
            pool[node].children[col] = child;
            node = child;
        }

        // 3. SIMULATION
        double value = simulate(b, rng);

        // 4. BACKUP
        backup(pool, node, value);
    }

    // Финальный ход — самый ПОСЕЩАЕМЫЙ корневой ребёнок (не с лучшим Q).
    int best_col = -1, best_n = -1;
    for (int col = 0; col < WIDTH; ++col) {
        int ch = pool[0].children[col];
        if (ch == -1) continue;
        if (pool[ch].N > best_n) {
            best_n   = pool[ch].N;
            best_col = col;
        }
    }
    return best_col;
}

// ---------------------------------------------------------------------------
// ВСПОМОГАТЕЛЬНОЕ: печать доски
// ---------------------------------------------------------------------------
static void print_board(const Board& b) {
    for (int row = HEIGHT - 1; row >= 0; --row) {
        for (int col = 0; col < WIDTH; ++col) {
            uint64_t bit = 1ULL << (col * H1 + row);
            if      (b.pos[0] & bit) std::cout << " X";
            else if (b.pos[1] & bit) std::cout << " O";
            else                     std::cout << " .";
        }
        std::cout << '\n';
    }
    std::cout << " 0 1 2 3 4 5 6\n";
}

int main() {
    std::mt19937 rng(42);
    Board b;

    const int ITERS = 20000;

    while (!is_terminal(b)) {
        int col = search(b, ITERS, rng);
        std::cout << "игрок " << b.player << " -> столбец " << col << "\n";
        play(b, col);
        print_board(b);
        std::cout << '\n';
    }

    if (wins(b.pos[0]))      std::cout << "победил X\n";
    else if (wins(b.pos[1])) std::cout << "победил O\n";
    else                     std::cout << "ничья\n";

    return 0;
}