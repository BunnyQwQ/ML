#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <limits>
#include <windows.h>

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
    m = p & (p >> 7);   if (m & (m >> 14)) return true;
    m = p & (p >> 1);   if (m & (m >> 2))  return true;
    m = p & (p >> 8);   if (m & (m >> 16)) return true;
    m = p & (p >> 6);   if (m & (m >> 12)) return true;
    return false;
}

inline bool can_play(const Board& b, int col) {
    return b.heights[col] < HEIGHT;
}

inline void play(Board& b, int col) {
    int ind = col * H1 + b.heights[col];
    b.pos[b.player] |= 1ULL << ind;
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
    double W    = 0.0;
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

        double q     = pool[ch].W / pool[ch].N;
        double score = -q + c_puct * std::sqrt(log_np / pool[ch].N);

        if (score > best_score) {
            best_score = score;
            best_idx   = ch;
        }
    }
    return best_idx;
}

static double simulate(Board b, std::mt19937& rng) {
    int me = b.player;
    while (true) {
        if (wins(b.pos[1 - b.player]))
            return ((1 - b.player) == me) ? 1.0 : -1.0;
        if (b.moves == CELLS) return 0.0;

        int legal[WIDTH], n = 0;
        for (int c = 0; c < WIDTH; ++c)
            if (can_play(b, c)) legal[n++] = c;

        std::uniform_int_distribution<int> pick(0, n - 1);
        play(b, legal[pick(rng)]);
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

struct SearchResult {
    int best_col = -1;
    std::array<int, WIDTH>    visits{};
    std::array<double, WIDTH> winrate{};
};

static SearchResult search(const Board& root_board, int iterations,
                           std::mt19937& rng, double c_puct = 1.41) {
    std::vector<Node> pool;
    pool.reserve(iterations + 1);
    make_node(pool, -1, -1, root_board);

    for (int it = 0; it < iterations; ++it) {
        Board b = root_board;
        int node = 0;

        while (pool[node].untried.empty() && !is_terminal(b)) {
            node = select_child(pool, node, c_puct);
            play(b, pool[node].move);
        }

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

        double value = simulate(b, rng);

        backup(pool, node, value);
    }

    SearchResult res;
    int best_n = -1;
    for (int col = 0; col < WIDTH; ++col) {
        int ch = pool[0].children[col];
        if (ch == -1) continue;

        res.visits[col]  = pool[ch].N;
        res.winrate[col] = -pool[ch].W / pool[ch].N;

        if (pool[ch].N > best_n) {
            best_n       = pool[ch].N;
            res.best_col = col;
        }
    }
    return res;
}

static void print_board(const Board& b) {
    std::cout << '\n';
    for (int row = HEIGHT - 1; row >= 0; --row) {
        std::cout << " |";
        for (int col = 0; col < WIDTH; ++col) {
            uint64_t bit = 1ULL << (col * H1 + row);
            if (b.pos[0] & bit)      std::cout << " X";
            else if (b.pos[1] & bit) std::cout << " O";
            else                     std::cout << " .";
        }
        std::cout << " |\n";
    }
    std::cout << " +---------------+\n";
    std::cout << "   0 1 2 3 4 5 6\n\n";
}

static void print_stats(const SearchResult& res, double seconds, int iterations) {
    std::cout << "  столбец:  ";
    for (int c = 0; c < WIDTH; ++c) std::cout << std::setw(7) << c;
    std::cout << "\n  посещений:";
    for (int c = 0; c < WIDTH; ++c) std::cout << std::setw(7) << res.visits[c];
    std::cout << "\n  винрейт:  ";
    for (int c = 0; c < WIDTH; ++c) {
        if (res.visits[c] > 0)
            std::cout << std::setw(7) << std::fixed << std::setprecision(2) << res.winrate[c];
        else
            std::cout << std::setw(7) << "-";
    }
    std::cout << "\n  " << iterations << " симуляций за "
              << std::fixed << std::setprecision(2) << seconds << " с  ("
              << static_cast<long long>(iterations / seconds) << " сим/с)\n\n";
}

static int read_human_move(const Board& b) {
    while (true) {
        std::cout << "твой ход (0-6, q — выход): ";
        std::string line;
        if (!std::getline(std::cin, line)) return -1;
        if (line == "q" || line == "Q") return -1;

        try {
            int col = std::stoi(line);
            if (col < 0 || col >= WIDTH) {
                std::cout << "  столбец вне диапазона\n";
                continue;
            }
            if (!can_play(b, col)) {
                std::cout << "  столбец заполнен\n";
                continue;
            }
            return col;
        } catch (...) {
            std::cout << "  нужно число от 0 до 6\n";
        }
    }
}

int main() {
    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);
    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "Connect Four против MCTS\n"
                 "X ходит первым, O вторым.\n\n";

    int human = 0;
    while (true) {
        std::cout << "играть за X (первым) или O (вторым)? [x/o]: ";
        std::string s;
        if (!std::getline(std::cin, s)) return 0;
        if (s == "x" || s == "X") { human = 0; break; }
        if (s == "o" || s == "O") { human = 1; break; }
    }

    int iterations = 20000;
    std::cout << "сила MCTS (число симуляций, Enter = 20000): ";
    {
        std::string s;
        std::getline(std::cin, s);
        if (!s.empty()) {
            try { iterations = std::max(100, std::stoi(s)); } catch (...) {}
        }
    }

    Board b;
    print_board(b);

    while (!is_terminal(b)) {
        int col;

        if (b.player == human) {
            col = read_human_move(b);
            if (col == -1) { std::cout << "выход\n"; return 0; }
        } else {
            auto t0 = std::chrono::steady_clock::now();
            SearchResult res = search(b, iterations, rng);
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();

            col = res.best_col;
            std::cout << "\nMCTS выбрал столбец " << col << "\n";
            print_stats(res, sec, iterations);
        }

        play(b, col);
        print_board(b);
    }

    if (wins(b.pos[0]))
        std::cout << (human == 0 ? "ты победил (X)\n" : "MCTS победил (X)\n");
    else if (wins(b.pos[1]))
        std::cout << (human == 1 ? "ты победил (O)\n" : "MCTS победил (O)\n");
    else
        std::cout << "ничья\n";

    return 0;
}