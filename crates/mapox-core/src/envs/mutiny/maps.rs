//! The ships a mutiny is played on, drawn by hand.
//!
//! Rows are written top first. `#` is hull, `.` deck, `S` a deck cell an
//! agent can start on, `M` the meeting button, `H` a traitor-only tile, and a
//! lowercase letter a chore button, the letter naming the room it belongs to:
//! every `a` on a ship is one room's set of buttons.
//! Adjacent `H` tiles can form a passage for traitors.

use serde::{Deserialize, Serialize};

/// Which ship a game is played on.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutinyMap {
    /// A different ship each game, picked by the episode's seed.
    #[default]
    Random,
    Frigate,
    Hauler,
    Station,
}

impl MutinyMap {
    /// The ship this names, or `None` for a random one.
    pub(super) fn ship(self) -> Option<&'static Ship> {
        match self {
            MutinyMap::Random => None,
            MutinyMap::Frigate => Some(&SHIPS[0]),
            MutinyMap::Hauler => Some(&SHIPS[1]),
            MutinyMap::Station => Some(&SHIPS[2]),
        }
    }
}

pub(super) struct Ship {
    /// What the tests call it when one of them fails.
    #[cfg(test)]
    pub name: &'static str,
    pub rows: &'static [&'static str],
}

impl Ship {
    pub fn width(&self) -> i32 {
        self.rows[0].len() as i32
    }

    pub fn height(&self) -> i32 {
        self.rows.len() as i32
    }
}

/// Most rooms any ship has, which bounds the letters a chore button uses.
pub(super) const MAX_ROOMS: usize = 8;

pub(super) const SHIPS: [Ship; 3] = [
    // Two decks of three rooms either side of a midships hall, joined by
    // two long corridors: every room is two turns from the meeting button.
    Ship {
        #[cfg(test)]
        name: "frigate",
        rows: &[
            "###############################",
            "#a.......a#b.......b#c.......c#",
            "#.........#.........#.........#",
            "#....a....#H...b....#....c...H#",
            "#.........#.........#.........#",
            "#.........#.........#.........#",
            "####..#########..#########..###",
            "#.............................#",
            "#.............................#",
            "####..####...........####..####",
            "#d......d#...SSSSS...#e......e#",
            "#........#...S.M.S...#........#",
            "#...d........SSSSS.........e..#",
            "#.............................#",
            "#d......H#...........#H......e#",
            "####..####...........####..####",
            "#.............................#",
            "#.............................#",
            "####..#########..#########..###",
            "#.........#.........#.........#",
            "#.........#.........#.........#",
            "#....f....#....g...H#....h....#",
            "#.........#.........#.........#",
            "#f.......f#g.......g#h.......h#",
            "###############################",
        ],
    },
    // One spine with four holds off each side. Everyone crosses the same
    // corridor, and the meeting button sits in the middle of it.
    Ship {
        #[cfg(test)]
        name: "hauler",
        rows: &[
            "#####################################",
            "#a......a#b......b#c......c#d......d#",
            "#........#........#........#........#",
            "#......a.#H.....b.#.c.....H#.d......#",
            "#........#........#........#........#",
            "###..########..#######..#######..####",
            "#...............SSSSS...............#",
            "#...............SSMSS...............#",
            "#...............SSSSS...............#",
            "###..########..#######..#######..####",
            "#........#........#........#........#",
            "#.e......#......f.#H.g.....#.....h.H#",
            "#........#........#........#........#",
            "#e......e#f......f#g......g#h......h#",
            "#####################################",
        ],
    },
    // A ring around a walled core. The four big rooms open off the ring
    // and the four corner rooms off the big ones, out of everyone's way.
    Ship {
        #[cfg(test)]
        name: "station",
        rows: &[
            "#############################",
            "#e...e##a...........a##f...f#",
            "#.....##.............##.....#",
            "#H..e.##......a......##.f..H#",
            "#.....##.............##.....#",
            "###.#########...#########.###",
            "###.###...............###.###",
            "#b...b#...............#c...c#",
            "#.....#..####...####..#.....#",
            "#.....#..#.........#..#.....#",
            "#..b..#..#..SSSSS..#..#..c..#",
            "#.....#..#..S...S..#..#.....#",
            "#...........S.M.S...........#",
            "#.....#..#..S...S..#..#.....#",
            "#..b..#..#..SSSSS..#..#..c..#",
            "#.....#..#.........#..#.....#",
            "#.....#..####...####..#.....#",
            "#b...b#...............#c...c#",
            "###.###...............###.###",
            "###.#########...#########.###",
            "#.....##.............##.....#",
            "#H..g.##......d......##.h..H#",
            "#.....##.............##.....#",
            "#g...g##d...........d##h...h#",
            "#############################",
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet, VecDeque};

    const NEIGHBOURS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    fn cell(ship: &Ship, x: i32, row: i32) -> u8 {
        if (0..ship.width()).contains(&x) && (0..ship.height()).contains(&row) {
            ship.rows[row as usize].as_bytes()[x as usize]
        } else {
            b'#'
        }
    }

    fn crew_walkable(tile: u8) -> bool {
        matches!(tile, b'.' | b'S')
    }

    fn cells(ship: &Ship) -> impl Iterator<Item = (i32, i32, u8)> + '_ {
        (0..ship.height())
            .flat_map(move |row| (0..ship.width()).map(move |x| (x, row, cell(ship, x, row))))
    }

    /// The tiles reachable from the first start cell with the given movement rules.
    fn reachable(ship: &Ship, walkable: impl Fn(u8) -> bool) -> HashSet<(i32, i32)> {
        let start = cells(ship)
            .find(|&(_, _, tile)| tile == b'S')
            .map(|(x, row, _)| (x, row))
            .expect("a ship has a start cell");
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some((x, row)) = queue.pop_front() {
            for (dx, dy) in NEIGHBOURS {
                let next = (x + dx, row + dy);
                if walkable(cell(ship, next.0, next.1)) && seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        seen
    }

    #[test]
    fn every_ship_is_a_walled_rectangle_of_known_tiles() {
        for ship in &SHIPS {
            for row in ship.rows {
                assert_eq!(row.len() as i32, ship.width(), "{} is ragged", ship.name);
            }
            for (x, row, tile) in cells(ship) {
                assert!(
                    matches!(tile, b'#' | b'.' | b'S' | b'M' | b'H' | b'a'..=b'h'),
                    "{} has an unknown tile {:?}",
                    ship.name,
                    tile as char
                );
                let edge = x == 0 || row == 0 || x == ship.width() - 1 || row == ship.height() - 1;
                assert!(!edge || tile == b'#', "{} leaks at {x},{row}", ship.name);
            }
        }
    }

    /// Everyone can reach the deck and buttons, and traitors can reach the hiding tiles.
    #[test]
    fn every_fitting_is_reachable_on_foot() {
        for ship in &SHIPS {
            let reached = reachable(ship, crew_walkable);
            let reached_by_traitors = reachable(ship, |tile| crew_walkable(tile) || tile == b'H');
            for (x, row, tile) in cells(ship) {
                if crew_walkable(tile) {
                    assert!(
                        reached.contains(&(x, row)),
                        "{} deck {x},{row} is cut off",
                        ship.name
                    );
                } else if tile == b'H' {
                    assert!(
                        reached_by_traitors.contains(&(x, row)),
                        "{} hiding tile {x},{row} is cut off",
                        ship.name
                    );
                } else if tile != b'#' {
                    assert!(
                        NEIGHBOURS
                            .iter()
                            .any(|(dx, dy)| reached.contains(&(x + dx, row + dy))),
                        "{} fitting {:?} at {x},{row} is walled in",
                        ship.name,
                        tile as char
                    );
                }
            }
        }
    }

    /// A full crew of ten has a start cell each, one button calls the vote,
    /// and a chore always has a second button to move on to.
    #[test]
    fn every_ship_has_what_a_game_needs() {
        for ship in &SHIPS {
            let mut counts: HashMap<u8, usize> = HashMap::new();
            for (_, _, tile) in cells(ship) {
                *counts.entry(tile).or_default() += 1;
            }

            assert!(counts[&b'S'] >= 10, "{} is short of start cells", ship.name);
            assert_eq!(counts[&b'M'], 1, "{} meeting buttons", ship.name);
            assert!(counts[&b'H'] >= 2, "{} is short of hiding spots", ship.name);

            let rooms: Vec<u8> = (b'a'..=b'h').filter(|r| counts.contains_key(r)).collect();
            assert!(rooms.len() >= 3 && rooms.len() <= MAX_ROOMS);
            // the letters run unbroken from `a`, so a letter is a room index
            assert_eq!(*rooms.last().unwrap(), b'a' + rooms.len() as u8 - 1);
            for room in rooms {
                assert!(
                    counts[&room] >= 2,
                    "{} room {:?} has one button",
                    ship.name,
                    room as char
                );
            }
        }
    }
}
