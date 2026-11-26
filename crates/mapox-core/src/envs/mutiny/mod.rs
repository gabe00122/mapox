//! Mutiny: a social deduction game on a ship. Most of the crew are loyal and
//! have chores to do; a few are traitors, who look the same and are out to
//! kill them.
//!
//! Each game is dealt afresh: a ship from [`maps`], a color for every crew
//! member, and who the traitors are. Color is all that tells the crew apart,
//! so it is shuffled every game, and an agent's role is shown to it alone, in
//! its UI band.
//!
//! Agents interact with fittings by walking into them:
//!
//! - **Chores.** Every loyal crew member has a queue of chores, each a room
//!   and a number of button presses. The button its chore wants next shows
//!   lit, to it alone; pressing it lights another button in the same room,
//!   and so on until the chore is done and the next one starts in another
//!   room. A pressed button shows as pressed to everyone for a step. Traitors
//!   have no chores and are shown no lit button, but any chore button a
//!   traitor walks into shows as pressed all the same, and counts for
//!   nothing: looking busy is theirs to work out.
//! - **The meeting button** calls a vote. A crew member can call only so
//!   many a game, and no one can for a while after a game starts or a vote
//!   ends; the button shows off to whoever cannot use it.
//! - **A corpse** calls a vote when anyone living walks into it.
//!
//! *Attack* lets a traitor kill every loyal crew member in sight within
//! `attack_radius` cells, in all directions at once, and then has to cool down.
//! A traitor can walk onto hiding tiles, which cover its sprite in other
//! agents' views. Adjacent hiding tiles can form a passage. Traitors can
//! still act normally from these tiles and block other traitors' movement.
//!
//! The dead leave a corpse where they fell and carry on as ghosts: unseen by
//! the living, free to drift through anything, and able to do nothing but
//! watch. A dead crew member's unfinished chores pass to the living.
//!
//! A vote takes everyone to the vote room, where no one can attack and the
//! living walk through each other. Each living crew member has a button
//! there in its color, and walking into one casts a vote to kick its owner,
//! shown at once in the count beside the button; there is a skip button too.
//! The vote ends when everyone living has voted or its time runs out.
//! Whoever has the most votes is kicked, unless that is a tie or no more
//! than the skips, and becomes a ghost without a corpse; kicking a traitor
//! pays the crew, and kicking a loyal crew member pays the traitors. Then
//! everyone goes back to where they started the game, the corpses are
//! cleared away, and the attack and meeting cooldowns start over.
//!
//! The crew win when the chores are done or when no traitor is left; the
//! traitors win when they are as many as the loyal crew still alive. A game
//! has no clock of its own, so waiting wins it for no one. A win pays
//! everyone on that side, dead or alive, everyone is flagged terminated, and
//! the next game starts at once; the episode itself runs until `length`, like
//! every env here, and a game it cuts short pays no one.
//!
//! The UI band's top row is the agent's role, then for a traitor its fellow
//! traitors as they look (as corpses once dead), and at the right the count
//! of loyal crew dead and of traitors dead. The row beneath is the button
//! presses the crew still owe, a traitor's attack cooldown, and at the
//! right, while a vote is on, the steps it has left.

mod maps;

use std::collections::VecDeque;

use ndarray::{Array2, ArrayViewMut2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

pub use maps::MutinyMap;
use maps::{MAX_ROOMS, SHIPS, Ship};

use crate::{
    env::Environment,
    envs::common::{
        Direction, Position, UI_HEIGHT,
        fov::{self, ViewTile},
        ui::write_number,
        vocab_enum::VocabEnum,
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_CREW, AGENT_CREW_GHOST, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP,
        PRIMARY_ACTION, TILE_CHORE_BUTTON, TILE_CHORE_BUTTON_LIT, TILE_CHORE_BUTTON_PRESSED,
        TILE_CORPSES, TILE_EMPTY, TILE_HIDING_SPOT, TILE_HIDING_SPOT_OCCUPIED, TILE_MASK,
        TILE_MEETING_BUTTON, TILE_MEETING_BUTTON_OFF, TILE_SKIP_BUTTON, TILE_UI, TILE_VOTE_BUTTONS,
        TILE_WALL, UI_ATTACK, UI_CHORES, UI_DEAD_CREW, UI_DEAD_TRAITORS, UI_DIGITS, UI_ROLE_CREW,
        UI_ROLE_TRAITOR, UI_VOTE,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

/// Most agents a game takes: one for each color.
pub const MAX_CREW: usize = AGENT_CREW.len();

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MutinyConfig {
    /// Everyone aboard, traitors included; at most [`MAX_CREW`].
    pub num_agents: usize,
    /// How many of them are dealt the traitor's role each game.
    pub num_traitors: usize,
    pub map: MutinyMap,

    pub view_width: i32,
    pub view_height: i32,

    /// Chores each crew member starts a game with, each in a different room
    /// while the ship has rooms to spare.
    pub chores_per_agent: u32,
    /// Button presses a chore takes.
    pub presses_per_chore: u32,

    /// How far an attack reaches, in cells, diagonals included.
    pub attack_radius: i32,
    /// Steps from one attack to the traitor's next, and from the start of a
    /// game or the end of a vote to its first.
    pub attack_cooldown: u32,

    /// Votes each crew member may call with the meeting button in a game.
    /// Walking into a corpse calls one regardless.
    pub meetings_per_agent: u32,
    /// Steps the meeting button stays off after a game starts or a vote ends.
    pub meeting_cooldown: u32,
    /// Steps a vote lasts, unless everyone living votes sooner.
    pub vote_steps: u32,

    /// Paid to a loyal crew member for each press of its lit button.
    pub chore_reward: f32,
    /// Paid to a traitor for each crew member its attack kills.
    pub kill_reward: f32,
    /// Paid to every loyal crew member, dead or alive, when a vote kicks a
    /// traitor.
    pub traitor_kicked_reward: f32,
    /// Paid to every traitor, dead or alive, when a vote kicks a loyal crew
    /// member.
    pub crew_kicked_reward: f32,
    /// Paid to everyone on the side that wins a game, and
    /// [`loss_reward`](Self::loss_reward) to everyone on the other.
    pub win_reward: f32,
    pub loss_reward: f32,
}

impl Default for MutinyConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            num_traitors: 2,
            map: MutinyMap::Random,
            view_width: 15,
            view_height: 15,
            chores_per_agent: 3,
            presses_per_chore: 4,
            attack_radius: 1,
            attack_cooldown: 30,
            meetings_per_agent: 1,
            meeting_cooldown: 20,
            vote_steps: 30,
            chore_reward: 0.05,
            kill_reward: 0.25,
            traitor_kicked_reward: 0.25,
            crew_kicked_reward: 0.25,
            win_reward: 1.0,
            loss_reward: -1.0,
        }
    }
}

vocab_enum!(MutinyObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    ChoreButton => TILE_CHORE_BUTTON,
    ChoreButtonLit => TILE_CHORE_BUTTON_LIT,
    ChoreButtonPressed => TILE_CHORE_BUTTON_PRESSED,
    MeetingButton => TILE_MEETING_BUTTON,
    MeetingButtonOff => TILE_MEETING_BUTTON_OFF,
    HidingSpot => TILE_HIDING_SPOT,
    HidingSpotOccupied => TILE_HIDING_SPOT_OCCUPIED,
    SkipButton => TILE_SKIP_BUTTON,
    Ghost => AGENT_CREW_GHOST,
    Crew0 => AGENT_CREW[0],
    Crew1 => AGENT_CREW[1],
    Crew2 => AGENT_CREW[2],
    Crew3 => AGENT_CREW[3],
    Crew4 => AGENT_CREW[4],
    Crew5 => AGENT_CREW[5],
    Crew6 => AGENT_CREW[6],
    Crew7 => AGENT_CREW[7],
    Crew8 => AGENT_CREW[8],
    Crew9 => AGENT_CREW[9],
    Corpse0 => TILE_CORPSES[0],
    Corpse1 => TILE_CORPSES[1],
    Corpse2 => TILE_CORPSES[2],
    Corpse3 => TILE_CORPSES[3],
    Corpse4 => TILE_CORPSES[4],
    Corpse5 => TILE_CORPSES[5],
    Corpse6 => TILE_CORPSES[6],
    Corpse7 => TILE_CORPSES[7],
    Corpse8 => TILE_CORPSES[8],
    Corpse9 => TILE_CORPSES[9],
    VoteButton0 => TILE_VOTE_BUTTONS[0],
    VoteButton1 => TILE_VOTE_BUTTONS[1],
    VoteButton2 => TILE_VOTE_BUTTONS[2],
    VoteButton3 => TILE_VOTE_BUTTONS[3],
    VoteButton4 => TILE_VOTE_BUTTONS[4],
    VoteButton5 => TILE_VOTE_BUTTONS[5],
    VoteButton6 => TILE_VOTE_BUTTONS[6],
    VoteButton7 => TILE_VOTE_BUTTONS[7],
    VoteButton8 => TILE_VOTE_BUTTONS[8],
    VoteButton9 => TILE_VOTE_BUTTONS[9],
    Digit0 => UI_DIGITS[0],
    Digit1 => UI_DIGITS[1],
    Digit2 => UI_DIGITS[2],
    Digit3 => UI_DIGITS[3],
    Digit4 => UI_DIGITS[4],
    Digit5 => UI_DIGITS[5],
    Digit6 => UI_DIGITS[6],
    Digit7 => UI_DIGITS[7],
    Digit8 => UI_DIGITS[8],
    Digit9 => UI_DIGITS[9],
    UiRoleCrew => UI_ROLE_CREW,
    UiRoleTraitor => UI_ROLE_TRAITOR,
    UiDeadCrew => UI_DEAD_CREW,
    UiDeadTraitors => UI_DEAD_TRAITORS,
    UiChores => UI_CHORES,
    UiAttack => UI_ATTACK,
    UiVote => UI_VOTE,
});

const DIGIT_TILES: [MutinyObs; 10] = [
    MutinyObs::Digit0,
    MutinyObs::Digit1,
    MutinyObs::Digit2,
    MutinyObs::Digit3,
    MutinyObs::Digit4,
    MutinyObs::Digit5,
    MutinyObs::Digit6,
    MutinyObs::Digit7,
    MutinyObs::Digit8,
    MutinyObs::Digit9,
];

impl MutinyObs {
    /// The tile `offset` places on from `first`, for the runs of ten the
    /// vocab keeps in color order.
    fn nth(first: MutinyObs, offset: u8) -> MutinyObs {
        MutinyObs::TABLE[first as usize + usize::from(offset)]
    }

    fn crew(color: u8) -> MutinyObs {
        Self::nth(MutinyObs::Crew0, color)
    }

    fn corpse(color: u8) -> MutinyObs {
        Self::nth(MutinyObs::Corpse0, color)
    }

    fn vote_button(color: u8) -> MutinyObs {
        Self::nth(MutinyObs::VoteButton0, color)
    }

    fn is_crew(self) -> bool {
        (MutinyObs::Crew0 as u16..=MutinyObs::Crew9 as u16).contains(&(self as u16))
    }

    fn is_corpse(self) -> bool {
        (MutinyObs::Corpse0 as u16..=MutinyObs::Corpse9 as u16).contains(&(self as u16))
    }
}

impl ViewTile for MutinyObs {
    const MASK: Self = MutinyObs::Mask;

    /// Only the hull blocks sight: the fittings, the dead and the vote
    /// room's boards are seen over.
    fn opaque(self) -> bool {
        self == MutinyObs::TileWall
    }
}

vocab_enum!(MutinyAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Noop => NOOP,
    Attack => PRIMARY_ACTION,
});

const STEPS: [Direction; 4] = [
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
];

impl MutinyAction {
    /// The cell-to-cell step a move action takes.
    fn step(self) -> Option<Position> {
        use MutinyAction::*;
        match self {
            MoveUp | MoveRight | MoveDown | MoveLeft => Some(STEPS[self as usize].to_pos()),
            Noop | Attack => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Crew,
    Traitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Alive,
    Dead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ballot {
    Skip,
    /// A vote to kick the agent with this id.
    Kick(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Play,
    Vote { steps_left: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chore {
    room: usize,
    presses_left: u32,
}

#[derive(Debug, Clone)]
struct Member {
    role: Role,
    color: u8,
    status: Status,
    position: Position,
    /// Where it starts the game, and comes back to after every vote.
    spawn: Position,
    /// The front chore is the one it is on.
    chores: VecDeque<Chore>,
    /// The button its chore wants pressed next.
    lit: Option<Position>,
    /// The button it pressed last, which the next one lit is never.
    pressed: Option<Position>,
    attack_cooldown: u32,
    meetings_left: u32,
    /// What it has voted in the vote that is on.
    ballot: Option<Ballot>,
}

impl Member {
    fn alive(&self) -> bool {
        self.status != Status::Dead
    }
}

/// The vote room, drawn top row first. `0`-`9` are the vote buttons by color,
/// `s` the skip button, `t` the count of votes for the button beside it, and
/// `_` a blank. Those make up a board along each side of the room, none of
/// which blocks sight, so every cell of the floor sees every button and
/// count.
const VOTE_ROOM: [&str; 10] = [
    "########", "#tttttt#", "#012345#", "#......#", "#......#", "#......#", "#......#", "#6789_s#",
    "#tttt_t#", "########",
];
const VOTE_WIDTH: i32 = VOTE_ROOM[0].len() as i32;
const VOTE_HEIGHT: i32 = VOTE_ROOM.len() as i32;
/// Rows of [`VOTE_ROOM`] the two banks of buttons are in, counted from its
/// bottom as map y is. Each bank's counts are the row behind it.
const VOTE_TOP_BANK: i32 = 7;
const VOTE_BOTTOM_BANK: i32 = 2;
/// Colors with a button in the top bank; the rest are in the bottom one.
const VOTE_TOP_COLORS: u8 = 6;
const VOTE_SKIP_X: i32 = 6;
const VOTE_FLOOR: std::ops::RangeInclusive<i32> = 3..=6;
/// Rows of the two boards, each a bank and the counts behind it.
const VOTE_BOARDS: [i32; 4] = [1, 2, 7, 8];

/// Where things are on the padded grid: the ship above, the vote room below.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    /// Hull around the interior, half a view each way, so a view window
    /// never leaves the grid.
    pad: Position,
    /// Size of the interior: the largest ship the config can deal, on top of
    /// the vote room.
    inner: Position,
    /// Tallest ship the config can deal.
    ship_height: i32,
    /// Bottom-left corner of the vote room.
    vote_origin: Position,
}

impl Geometry {
    fn new(config: &MutinyConfig) -> Self {
        let ships = config.map.ship().map_or(&SHIPS[..], std::slice::from_ref);
        let ship_width = ships.iter().map(Ship::width).max().unwrap();
        let ship_height = ships.iter().map(Ship::height).max().unwrap();

        let pad = Position::new(config.view_width / 2, config.view_height / 2);
        let inner = Position::new(ship_width.max(VOTE_WIDTH), ship_height + VOTE_HEIGHT);
        Self {
            pad,
            inner,
            ship_height,
            vote_origin: pad + Position::new((inner.x - VOTE_WIDTH) / 2, 0),
        }
    }

    fn grid(&self) -> (usize, usize) {
        let size = self.inner + self.pad * 2;
        (size.x as usize, size.y as usize)
    }

    /// Whether `position` is in the interior, hull and all: where a ghost
    /// can drift.
    fn inside(&self, position: Position) -> bool {
        let local = position - self.pad;
        (0..self.inner.x).contains(&local.x) && (0..self.inner.y).contains(&local.y)
    }

    /// The grid cell of column `x` of row `row` of `ship`'s drawing. The
    /// ship is centred in the space the largest one fills.
    fn ship_cell(&self, ship: &Ship, x: i32, row: i32) -> Position {
        let left = (self.inner.x - ship.width()) / 2;
        let bottom = VOTE_HEIGHT + (self.ship_height - ship.height()) / 2;
        self.pad + Position::new(left + x, bottom + ship.height() - 1 - row)
    }

    fn vote_button(&self, color: u8) -> Position {
        let (x, y) = if color < VOTE_TOP_COLORS {
            (1 + i32::from(color), VOTE_TOP_BANK)
        } else {
            (1 + i32::from(color - VOTE_TOP_COLORS), VOTE_BOTTOM_BANK)
        };
        self.vote_origin + Position::new(x, y)
    }

    fn skip_button(&self) -> Position {
        self.vote_origin + Position::new(VOTE_SKIP_X, VOTE_BOTTOM_BANK)
    }

    /// The count for a vote or skip `button`: the cell behind it.
    fn tally(&self, button: Position) -> Position {
        let behind = if button.y - self.vote_origin.y == VOTE_TOP_BANK {
            1
        } else {
            -1
        };
        button + Position::new(0, behind)
    }

    /// The `index`th cell of the vote room's floor, filling it row by row.
    fn vote_floor(&self, index: usize) -> Position {
        let width = VOTE_WIDTH - 2;
        let index = index as i32;
        self.vote_origin + Position::new(1 + index % width, VOTE_FLOOR.start() + index / width)
    }
}

#[derive(Debug, Clone)]
struct MutinyState {
    rng: SmallRng,
    members: Vec<Member>,
    order: Vec<usize>, // agent turn order
    time: usize,       // steps into the episode
    game_time: u32,    // steps into the game
    phase: Phase,

    base_map: Array2<MutinyObs>, // the ship and its fittings
    /// The base map with living agents, corpses, recent chore presses, and
    /// the vote room's buttons and counts. Hiding tiles cover their occupants
    /// when drawing observations.
    map: Array2<MutinyObs>,

    rooms: Vec<Vec<Position>>, // chore buttons by room; the ship's rooms come first
    num_rooms: usize,
    meeting_button: Position,
    spawns: Vec<Position>,
    by_color: [Option<usize>; MAX_CREW], // who wears each color this game

    corpses: Vec<Position>,
    pressed: Vec<Position>, // chore buttons pressed on the last step
    slain: Vec<usize>,      // killed so far this step
    called: bool,           // a vote was called this step
    meeting_cooldown: u32,
    /// Presses the loyal crew still owe between them.
    chores_left: u32,
    dead_crew: u32,
    dead_traitors: u32,
}

#[derive(Debug, Default, Clone)]
struct MutinyMetrics {
    games: f64,
    crew_reward: f64,
    traitor_reward: f64,
    wins_by_chores: f64,
    wins_by_vote: f64,
    traitor_wins: f64,
    game_steps: f64,
    presses: f64,
    kills: f64,
    meetings: f64,
    kicked_crew: f64,
    kicked_traitors: f64,
}

/// How a game ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    /// The crew pressed every button they owed.
    ChoresDone,
    /// No traitor is left.
    TraitorsGone,
    /// The traitors are as many as the loyal crew alive.
    Overrun,
}

impl Ending {
    fn winner(self) -> Role {
        match self {
            Ending::ChoresDone | Ending::TraitorsGone => Role::Crew,
            Ending::Overrun => Role::Traitor,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mutiny {
    pub config: MutinyConfig,
    state: MutinyState,
    metrics: MutinyMetrics,

    // max steps for a single episode
    length: usize,
    geometry: Geometry,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

/// Overwrites the cell of `view` showing `position`, if the viewer can see
/// it. `origin` is the map position of the view's first cell.
fn overlay(
    view: &mut ArrayViewMut2<'_, VocabId>,
    origin: Position,
    position: Position,
    tile: MutinyObs,
) {
    let cell = position - origin;
    let (width, height) = view.dim();
    if (0..width as i32).contains(&cell.x)
        && (0..height as i32).contains(&cell.y)
        && view[cell.idx()] != VocabId::from(MutinyObs::Mask)
    {
        view[cell.idx()] = tile.into();
    }
}

impl Mutiny {
    pub fn new(config: &MutinyConfig, length: usize) -> Self {
        assert!(
            (1..=MAX_CREW).contains(&config.num_agents),
            "mutiny takes 1 to {MAX_CREW} agents, one for each color"
        );
        assert!(
            config.num_traitors <= config.num_agents,
            "there are more traitors than agents"
        );
        assert!(
            config.chores_per_agent > 0 && config.presses_per_chore > 0,
            "a crew member needs a chore to do"
        );
        // the band's rows: role, fellow traitors, then the two death counts;
        // and chores, attack cooldown, then a vote's steps left
        let band = (6 + config.num_traitors as i32).max(12);
        assert!(
            config.view_width >= band,
            "the UI band needs a view at least {band} wide"
        );
        let geometry = Geometry::new(config);
        assert!(
            (0..=geometry.pad.x.min(geometry.pad.y)).contains(&config.attack_radius),
            "an attack reaches no further than half a view"
        );

        let action_vocab = MutinyAction::vocab();
        let obs_vocab = MutinyObs::vocab();
        let view_height = config.view_height + UI_HEIGHT as i32;
        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),
            state: MutinyState {
                rng: SmallRng::seed_from_u64(0),
                members: Vec::with_capacity(config.num_agents),
                order: (0..config.num_agents).collect(),
                time: 0,
                game_time: 0,
                phase: Phase::Play,
                base_map: Array2::from_elem(geometry.grid(), MutinyObs::TileWall),
                map: Array2::from_elem(geometry.grid(), MutinyObs::TileWall),
                rooms: vec![Vec::new(); MAX_ROOMS],
                num_rooms: 0,
                meeting_button: Position::default(),
                spawns: Vec::new(),
                by_color: [None; MAX_CREW],
                corpses: Vec::new(),
                pressed: Vec::new(),
                slain: Vec::new(),
                called: false,
                meeting_cooldown: 0,
                chores_left: 0,
                dead_crew: 0,
                dead_traitors: 0,
            },
            metrics: MutinyMetrics::default(),
            length,
            geometry,
            obs_spec,
            action_spec,
            obs_vocab,
            action_vocab,
        }
    }

    /// Deals a new game: a ship, and everyone's color, role, start cell and
    /// chores.
    fn new_game(&mut self) {
        let ship = match self.config.map.ship() {
            Some(ship) => ship,
            None => &SHIPS[self.state.rng.random_range(0..SHIPS.len())],
        };
        self.lay_out(ship);

        let state = &mut self.state;
        let num_agents = self.config.num_agents;

        let mut colors: [u8; MAX_CREW] = std::array::from_fn(|color| color as u8);
        colors.shuffle(&mut state.rng);
        // the first few of a shuffle of the crew are the traitors
        let mut dealt: Vec<usize> = (0..num_agents).collect();
        dealt.shuffle(&mut state.rng);
        let traitors = &dealt[..self.config.num_traitors];
        state.spawns.shuffle(&mut state.rng);

        state.by_color = [None; MAX_CREW];
        state.members.clear();
        for id in 0..num_agents {
            let spawn = state.spawns[id];
            state.by_color[usize::from(colors[id])] = Some(id);
            state.members.push(Member {
                role: if traitors.contains(&id) {
                    Role::Traitor
                } else {
                    Role::Crew
                },
                color: colors[id],
                status: Status::Alive,
                position: spawn,
                spawn,
                chores: VecDeque::new(),
                lit: None,
                pressed: None,
                attack_cooldown: self.config.attack_cooldown,
                meetings_left: self.config.meetings_per_agent,
                ballot: None,
            });
        }

        // each crew member's chores are in different rooms, until the ship
        // runs out of rooms and the list goes round again
        let mut rooms: Vec<usize> = (0..state.num_rooms).collect();
        state.chores_left = 0;
        for member in &mut state.members {
            if member.role != Role::Crew {
                continue;
            }
            rooms.shuffle(&mut state.rng);
            for chore in 0..self.config.chores_per_agent as usize {
                member.chores.push_back(Chore {
                    room: rooms[chore % rooms.len()],
                    presses_left: self.config.presses_per_chore,
                });
            }
            state.chores_left += self.config.chores_per_agent * self.config.presses_per_chore;
        }

        state.game_time = 0;
        state.phase = Phase::Play;
        state.corpses.clear();
        state.pressed.clear();
        state.slain.clear();
        state.called = false;
        state.meeting_cooldown = self.config.meeting_cooldown;
        state.dead_crew = 0;
        state.dead_traitors = 0;

        for id in 0..num_agents {
            self.light_next(id);
        }
        self.paint_members();
        self.paint_vote_panel();
    }

    /// Draws `ship` and the vote room's floor onto the base map, and notes
    /// where the ship's fittings are.
    fn lay_out(&mut self, ship: &Ship) {
        let state = &mut self.state;
        state.base_map.fill(MutinyObs::TileWall);
        state.spawns.clear();
        for room in &mut state.rooms {
            room.clear();
        }

        for (row, line) in ship.rows.iter().enumerate() {
            for (x, tile) in line.bytes().enumerate() {
                let position = self.geometry.ship_cell(ship, x as i32, row as i32);
                state.base_map[position.idx()] = match tile {
                    b'.' => MutinyObs::TileEmpty,
                    b'S' => {
                        state.spawns.push(position);
                        MutinyObs::TileEmpty
                    }
                    b'M' => {
                        state.meeting_button = position;
                        MutinyObs::MeetingButton
                    }
                    b'H' => MutinyObs::HidingSpot,
                    b'a'..=b'h' => {
                        state.rooms[usize::from(tile - b'a')].push(position);
                        MutinyObs::ChoreButton
                    }
                    _ => MutinyObs::TileWall,
                };
            }
        }
        state.num_rooms = state.rooms.iter().filter(|room| !room.is_empty()).count();

        for index in 0..((VOTE_WIDTH - 2) * VOTE_FLOOR.count() as i32) as usize {
            state.base_map[self.geometry.vote_floor(index).idx()] = MutinyObs::TileEmpty;
        }
        // the boards are blank until a vote sets them out
        for y in VOTE_BOARDS {
            for x in 1..VOTE_WIDTH - 1 {
                let cell = self.geometry.vote_origin + Position::new(x, y);
                state.base_map[cell.idx()] = MutinyObs::UI;
            }
        }
        state.map.assign(&state.base_map);
    }

    /// Draws every living agent onto the map.
    fn paint_members(&mut self) {
        let state = &mut self.state;
        for member in &state.members {
            if member.status == Status::Alive {
                state.map[member.position.idx()] = MutinyObs::crew(member.color);
            }
        }
    }

    /// Lifts every living agent off the map.
    fn erase_members(&mut self) {
        let state = &mut self.state;
        for member in &state.members {
            if member.status == Status::Alive {
                state.map[member.position.idx()] = state.base_map[member.position.idx()];
            }
        }
    }

    /// Sets out the vote room's buttons with their counts at zero: a button
    /// for each living crew member, the corpse of each dead one where its
    /// button was, and a blank for the colors no one wears.
    fn paint_vote_panel(&mut self) {
        let state = &mut self.state;
        for color in 0..MAX_CREW as u8 {
            let button = self.geometry.vote_button(color);
            let (tile, count) = match state.by_color[usize::from(color)] {
                Some(id) if state.members[id].alive() => {
                    (MutinyObs::vote_button(color), MutinyObs::Digit0)
                }
                Some(_) => (MutinyObs::corpse(color), MutinyObs::UI),
                None => (MutinyObs::UI, MutinyObs::UI),
            };
            state.map[button.idx()] = tile;
            state.map[self.geometry.tally(button).idx()] = count;
        }
        let skip = self.geometry.skip_button();
        state.map[skip.idx()] = MutinyObs::SkipButton;
        state.map[self.geometry.tally(skip).idx()] = MutinyObs::Digit0;
    }

    /// Lights the button `id`'s front chore wants next: any button in the
    /// chore's room but the one it just pressed.
    fn light_next(&mut self, id: usize) {
        let state = &mut self.state;
        let member = &mut state.members[id];
        member.lit = member.chores.front().map(|chore| {
            let pressed = member.pressed;
            let mut buttons = state.rooms[chore.room]
                .iter()
                .filter(|&&button| Some(button) != pressed);
            let pick = state.rng.random_range(0..buttons.clone().count());
            *buttons.nth(pick).expect("a room has a second button")
        });
    }

    /// Whether `id` could call a vote with the meeting button.
    fn can_call(&self, id: usize) -> bool {
        let member = &self.state.members[id];
        self.state.phase == Phase::Play
            && self.state.meeting_cooldown == 0
            && member.status == Status::Alive
            && member.meetings_left > 0
    }

    /// Calls `strike` with every loyal crew member an attack by `id` would
    /// kill: those in the open within `attack_radius` cells that `id` has a
    /// line of sight to. No one, unless `id` is a traitor able to attack.
    fn for_each_victim(&self, id: usize, mut strike: impl FnMut(usize)) {
        let state = &self.state;
        let attacker = &state.members[id];
        if state.phase != Phase::Play
            || attacker.role != Role::Traitor
            || attacker.status != Status::Alive
            || attacker.attack_cooldown > 0
        {
            return;
        }

        let radius = self.config.attack_radius;
        let reach = (2 * radius + 1) as usize;
        let center = Position::new(radius, radius);
        let origin = attacker.position - center;
        fov::shadowcast(
            center,
            (reach, reach),
            |cell| !state.map[(origin + cell).idx()].opaque(),
            |cell| {
                let position = origin + cell;
                if !state.map[position.idx()].is_crew() {
                    return;
                }
                let victim = state.members.iter().position(|member| {
                    member.status == Status::Alive
                        && member.role == Role::Crew
                        && member.position == position
                });
                if let Some(victim) = victim {
                    strike(victim);
                }
            },
        );
    }

    /// Passes a dead crew member's unfinished chores to the living, each to
    /// whoever then owes the fewest presses.
    fn hand_on_chores(&mut self, id: usize) {
        let chores = std::mem::take(&mut self.state.members[id].chores);
        self.state.members[id].lit = None;
        for chore in chores {
            let heir = self
                .state
                .members
                .iter()
                .enumerate()
                .filter(|(_, member)| member.role == Role::Crew && member.alive())
                .min_by_key(|(_, member)| {
                    member
                        .chores
                        .iter()
                        .map(|chore| chore.presses_left)
                        .sum::<u32>()
                })
                .map(|(heir, _)| heir);
            // with no one left to inherit, the game is over anyway
            let Some(heir) = heir else { break };

            self.state.members[heir].chores.push_back(chore);
            if self.state.members[heir].lit.is_none() {
                self.light_next(heir);
            }
        }
    }

    /// Makes a ghost of `id`, which leaves the map to the caller.
    fn die(&mut self, id: usize) {
        let member = &mut self.state.members[id];
        member.status = Status::Dead;
        match member.role {
            Role::Crew => self.state.dead_crew += 1,
            Role::Traitor => self.state.dead_traitors += 1,
        }
        self.hand_on_chores(id);
    }

    /// A ghost's move: one cell any way, through whatever is there.
    fn drift(&mut self, id: usize, step: Position) {
        let target = self.state.members[id].position + step;
        if self.geometry.inside(target) {
            self.state.members[id].position = target;
        }
    }

    /// A living agent's move into `target` outside a vote: a step onto open
    /// deck or an accessible hiding tile, or an interaction with a fitting.
    fn walk(&mut self, id: usize, target: Position, timestep: &mut TimeStepMut) {
        let state = &mut self.state;
        let member = &mut state.members[id];
        let tile = state.map[target.idx()];

        if tile == MutinyObs::TileEmpty
            || (tile == MutinyObs::HidingSpot && member.role == Role::Traitor)
        {
            state.map[member.position.idx()] = state.base_map[member.position.idx()];
            member.position = target;
            state.map[target.idx()] = MutinyObs::crew(member.color);
        } else if member.lit == Some(target) {
            let chore = member.chores.front_mut().expect("a lit button has a chore");
            chore.presses_left -= 1;
            if chore.presses_left == 0 {
                member.chores.pop_front();
            }
            member.pressed = Some(target);
            state.chores_left -= 1;
            timestep.reward[id] += self.config.chore_reward;
            self.metrics.crew_reward += f64::from(self.config.chore_reward);
            self.metrics.presses += 1.0;
            state.map[target.idx()] = MutinyObs::ChoreButtonPressed;
            state.pressed.push(target);
            self.light_next(id);
        } else if tile == MutinyObs::ChoreButton && member.role == Role::Traitor {
            // a traitor's press is only for show
            state.map[target.idx()] = MutinyObs::ChoreButtonPressed;
            state.pressed.push(target);
        } else if tile == MutinyObs::MeetingButton {
            if self.can_call(id) {
                self.state.members[id].meetings_left -= 1;
                self.state.called = true;
            }
        } else if tile.is_corpse() {
            state.called = true;
        }
    }

    /// A traitor's attack: everyone it reaches dies where they stand.
    fn attack(&mut self, id: usize, timestep: &mut TimeStepMut) {
        let mut victims = [false; MAX_CREW];
        self.for_each_victim(id, |victim| victims[victim] = true);

        for victim in (0..self.config.num_agents).filter(|&victim| victims[victim]) {
            let member = &self.state.members[victim];
            let (position, color) = (member.position, member.color);
            self.state.map[position.idx()] = MutinyObs::corpse(color);
            self.state.corpses.push(position);
            self.state.slain.push(victim);
            self.die(victim);

            self.state.members[id].attack_cooldown = self.config.attack_cooldown;
            timestep.reward[id] += self.config.kill_reward;
            self.metrics.traitor_reward += f64::from(self.config.kill_reward);
            self.metrics.kills += 1.0;
        }
    }

    /// Carries out one agent's action outside a vote.
    fn act(&mut self, id: usize, action: MutinyAction, timestep: &mut TimeStepMut) {
        let member = &self.state.members[id];
        let (status, position) = (member.status, member.position);

        match (status, action.step()) {
            (Status::Dead, Some(step)) => self.drift(id, step),
            (Status::Alive, Some(step)) => self.walk(id, position + step, timestep),
            (Status::Alive, None) if action == MutinyAction::Attack => self.attack(id, timestep),
            _ => {}
        }
    }

    /// One step outside a vote.
    fn step_play(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        // last step's presses have been seen
        while let Some(button) = self.state.pressed.pop() {
            self.state.map[button.idx()] = self.state.base_map[button.idx()];
        }

        for turn in 0..self.state.order.len() {
            let id = self.state.order[turn];
            // whoever was killed earlier in the step chose its action alive
            if !self.state.slain.contains(&id) {
                self.act(id, MutinyAction::from_id(actions[id]), timestep);
            }
        }
        self.state.slain.clear();

        // The cooldowns run down after the actions, so the count an agent
        // is shown is the number of steps it still has to wait.
        self.state.meeting_cooldown = self.state.meeting_cooldown.saturating_sub(1);
        for member in &mut self.state.members {
            member.attack_cooldown = member.attack_cooldown.saturating_sub(1);
        }

        if std::mem::take(&mut self.state.called) {
            self.start_vote();
        }
    }

    /// Clears the ship and takes everyone to the vote room.
    fn start_vote(&mut self) {
        self.erase_members();
        let state = &mut self.state;
        for corpse in state.corpses.drain(..) {
            state.map[corpse.idx()] = state.base_map[corpse.idx()];
        }
        for button in state.pressed.drain(..) {
            state.map[button.idx()] = state.base_map[button.idx()];
        }

        for (id, member) in state.members.iter_mut().enumerate() {
            member.position = self.geometry.vote_floor(id);
            member.ballot = None;
        }
        state.phase = Phase::Vote {
            steps_left: self.config.vote_steps,
        };
        self.metrics.meetings += 1.0;

        self.paint_vote_panel();
        self.paint_members();
    }

    /// What walking into `target` in the vote room votes for.
    fn ballot_at(&self, target: Position) -> Option<Ballot> {
        if target == self.geometry.skip_button() {
            return Some(Ballot::Skip);
        }
        (0..MAX_CREW as u8)
            .find(|&color| self.geometry.vote_button(color) == target)
            .and_then(|color| self.state.by_color[usize::from(color)])
            .map(Ballot::Kick)
    }

    /// Votes cast so far in the vote that is on, for `ballot`.
    fn votes_for(&self, ballot: Ballot) -> u32 {
        self.state
            .members
            .iter()
            .filter(|member| member.ballot == Some(ballot))
            .count() as u32
    }

    /// A living agent's move into `target` during a vote: a step onto the
    /// floor, whoever else is on it, or a vote cast with the button in the
    /// way. A vote is cast once, and not for the dead or for oneself.
    fn walk_vote(&mut self, id: usize, target: Position) {
        if self.state.map[target.idx()] == MutinyObs::TileEmpty {
            self.state.members[id].position = target;
            return;
        }

        let Some(ballot) = self.ballot_at(target) else {
            return;
        };
        let legal = match ballot {
            Ballot::Skip => true,
            Ballot::Kick(other) => other != id && self.state.members[other].alive(),
        };
        if self.state.members[id].ballot.is_some() || !legal {
            return;
        }

        self.state.members[id].ballot = Some(ballot);
        let count = self.votes_for(ballot).min(9) as usize;
        let tally = self.geometry.tally(target);
        self.state.map[tally.idx()] = DIGIT_TILES[count];
    }

    /// One step of a vote.
    fn step_vote(&mut self, actions: &[VocabId], steps_left: u32, timestep: &mut TimeStepMut) {
        // the living stand wherever they like in here, so they come off the
        // map while they move and go back on where they end up
        self.erase_members();
        for turn in 0..self.state.order.len() {
            let id = self.state.order[turn];
            let member = &self.state.members[id];
            let Some(step) = MutinyAction::from_id(actions[id]).step() else {
                continue;
            };
            if member.alive() {
                self.walk_vote(id, member.position + step);
            } else {
                self.drift(id, step);
            }
        }
        self.paint_members();

        let steps_left = steps_left - 1;
        let everyone_voted = self
            .state
            .members
            .iter()
            .all(|member| !member.alive() || member.ballot.is_some());
        if steps_left == 0 || everyone_voted {
            self.end_vote(timestep);
        } else {
            self.state.phase = Phase::Vote { steps_left };
        }
    }

    /// Counts the vote, kicks whoever lost it and pays the other side for
    /// it, and sends everyone back to where they started the game.
    fn end_vote(&mut self, timestep: &mut TimeStepMut) {
        let num_agents = self.config.num_agents;
        let votes: Vec<u32> = (0..num_agents)
            .map(|id| self.votes_for(Ballot::Kick(id)))
            .collect();
        let most = votes.iter().copied().max().unwrap_or(0);
        let leaders = votes.iter().filter(|&&count| count == most).count();

        self.erase_members();
        if leaders == 1 && most > self.votes_for(Ballot::Skip) {
            let kicked = votes.iter().position(|&count| count == most).unwrap();
            match self.state.members[kicked].role {
                Role::Crew => {
                    self.metrics.kicked_crew += 1.0;
                    self.pay(Role::Traitor, self.config.crew_kicked_reward, timestep);
                }
                Role::Traitor => {
                    self.metrics.kicked_traitors += 1.0;
                    self.pay(Role::Crew, self.config.traitor_kicked_reward, timestep);
                }
            }
            self.die(kicked);
        }

        for member in &mut self.state.members {
            member.position = member.spawn;
            member.ballot = None;
            member.attack_cooldown = self.config.attack_cooldown;
        }
        self.state.meeting_cooldown = self.config.meeting_cooldown;
        self.state.phase = Phase::Play;
        // the boards go back to blank counts, less whoever was kicked
        self.paint_vote_panel();
        self.paint_members();
    }

    /// How the game has ended, if it has.
    fn ending(&self) -> Option<Ending> {
        let living = |role: Role| {
            self.state
                .members
                .iter()
                .filter(|member| member.role == role && member.alive())
                .count()
        };
        let (crew, traitors) = (living(Role::Crew), living(Role::Traitor));

        if traitors == 0 {
            Some(Ending::TraitorsGone)
        } else if traitors >= crew {
            Some(Ending::Overrun)
        } else if self.state.chores_left == 0 {
            Some(Ending::ChoresDone)
        } else {
            None
        }
    }

    /// Pays `reward` to everyone on `role`'s side, dead or alive.
    fn pay(&mut self, role: Role, reward: f32, timestep: &mut TimeStepMut) {
        for (id, member) in self.state.members.iter().enumerate() {
            if member.role != role {
                continue;
            }
            timestep.reward[id] += reward;
            match role {
                Role::Crew => self.metrics.crew_reward += f64::from(reward),
                Role::Traitor => self.metrics.traitor_reward += f64::from(reward),
            }
        }
    }

    /// Pays both sides for a game that ended in `ending`.
    fn pay_out(&mut self, ending: Ending, timestep: &mut TimeStepMut) {
        let (winner, loser) = match ending.winner() {
            Role::Crew => (Role::Crew, Role::Traitor),
            Role::Traitor => (Role::Traitor, Role::Crew),
        };
        self.pay(winner, self.config.win_reward, timestep);
        self.pay(loser, self.config.loss_reward, timestep);

        self.metrics.games += 1.0;
        self.metrics.game_steps += f64::from(self.state.game_time);
        *match ending {
            Ending::ChoresDone => &mut self.metrics.wins_by_chores,
            Ending::TraitorsGone => &mut self.metrics.wins_by_vote,
            Ending::Overrun => &mut self.metrics.traitor_wins,
        } += 1.0;
    }

    /// Draws what `id` sees of the map into `view`.
    fn draw_view(&self, id: usize, view: &mut ArrayViewMut2<'_, VocabId>) {
        let state = &self.state;
        let member = &state.members[id];
        let (width, height) = view.dim();
        let center = Position::new(width as i32 / 2, height as i32 / 2);
        let origin = member.position - center;

        if member.alive() {
            fov::observe(&state.map, member.position, view);
        } else {
            // a ghost sees through the hull
            let window = fov::window(&state.map, member.position, (width, height));
            view.zip_mut_with(&window, |cell, &tile| *cell = tile.into());
        }

        for other in &state.members {
            let position = other.position;
            if other.alive() && state.base_map[position.idx()] == MutinyObs::HidingSpot {
                overlay(view, origin, position, MutinyObs::HidingSpot);
            } else if !member.alive()
                && !other.alive()
                && matches!(
                    state.map[position.idx()],
                    MutinyObs::TileEmpty | MutinyObs::TileWall
                )
            {
                // a ghost over a corpse or a fitting leaves that showing
                overlay(view, origin, position, MutinyObs::Ghost);
            }
        }
        if !member.alive() {
            view[center.idx()] = MutinyObs::Ghost.into();
            return;
        }

        if let Some(button) = member.lit {
            overlay(view, origin, button, MutinyObs::ChoreButtonLit);
        }
        if !self.can_call(id) {
            overlay(
                view,
                origin,
                state.meeting_button,
                MutinyObs::MeetingButtonOff,
            );
        }
        // Show the viewer its own hiding spot or crew color. In the vote
        // room someone else may be drawn on the same cell.
        view[center.idx()] = if state.base_map[member.position.idx()] == MutinyObs::HidingSpot {
            MutinyObs::HidingSpotOccupied
        } else {
            MutinyObs::crew(member.color)
        }
        .into();
    }

    /// Draws `id`'s UI band.
    fn draw_band(&self, id: usize, band: &mut ArrayViewMut2<'_, VocabId>) {
        let state = &self.state;
        let member = &state.members[id];
        let width = band.dim().0;
        let (top, bottom) = (UI_HEIGHT - 1, 0);
        band.fill(MutinyObs::UI.into());

        band[[0, top]] = match member.role {
            Role::Crew => MutinyObs::UiRoleCrew,
            Role::Traitor => MutinyObs::UiRoleTraitor,
        }
        .into();
        if member.role == Role::Traitor {
            let fellows = state
                .members
                .iter()
                .enumerate()
                .filter(|&(other, fellow)| other != id && fellow.role == Role::Traitor);
            for (slot, (_, fellow)) in fellows.enumerate() {
                band[[1 + slot, top]] = if fellow.alive() {
                    MutinyObs::crew(fellow.color)
                } else {
                    MutinyObs::corpse(fellow.color)
                }
                .into();
            }
        }
        band[[width - 6, top]] = MutinyObs::UiDeadCrew.into();
        write_number(
            band.slice_mut(s![width - 5..width - 3, top]),
            state.dead_crew,
            &DIGIT_TILES,
        );
        band[[width - 3, top]] = MutinyObs::UiDeadTraitors.into();
        write_number(
            band.slice_mut(s![width - 2.., top]),
            state.dead_traitors,
            &DIGIT_TILES,
        );

        band[[0, bottom]] = MutinyObs::UiChores.into();
        write_number(
            band.slice_mut(s![1..4, bottom]),
            state.chores_left,
            &DIGIT_TILES,
        );
        if member.role == Role::Traitor {
            band[[5, bottom]] = MutinyObs::UiAttack.into();
            write_number(
                band.slice_mut(s![6..8, bottom]),
                member.attack_cooldown,
                &DIGIT_TILES,
            );
        }
        if let Phase::Vote { steps_left } = state.phase {
            band[[width - 4, bottom]] = MutinyObs::UiVote.into();
            write_number(
                band.slice_mut(s![width - 3.., bottom]),
                steps_left,
                &DIGIT_TILES,
            );
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        for id in 0..self.config.num_agents {
            // hull padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![id, .., ..fov_height, 0]);
            self.draw_view(id, &mut view);
            let mut band = timestep.obs.slice_mut(s![id, .., fov_height.., 0]);
            self.draw_band(id, &mut band);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    /// Moves and the noop are always legal: a move into a fitting is how it
    /// is worked. Attack is legal for a traitor with someone to kill.
    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        timestep.action_mask.fill(true);
        for id in 0..self.config.num_agents {
            let mut in_reach = false;
            self.for_each_victim(id, |_| in_reach = true);

            timestep.action_mask[[id, MutinyAction::Attack as usize]] = in_reach;
        }
    }
}

impl Environment for Mutiny {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rng = SmallRng::seed_from_u64(seed);
        for (turn, id) in self.state.order.iter_mut().enumerate() {
            *id = turn;
        }
        self.state.time = 0;
        self.new_game();

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        timestep.reward.fill(0.0);
        for (last, &action) in timestep.last_action.iter_mut().zip(actions) {
            *last = action;
        }
        self.state.order.shuffle(&mut self.state.rng);

        match self.state.phase {
            Phase::Play => self.step_play(actions, timestep),
            Phase::Vote { steps_left } => self.step_vote(actions, steps_left, timestep),
        }
        self.state.time += 1;
        self.state.game_time += 1;

        // a game that ends is paid out on its last step, and the next is
        // dealt before that step is drawn
        let ending = self.ending();
        if let Some(ending) = ending {
            self.pay_out(ending, timestep);
            self.new_game();
        }

        self.encode_observations(timestep);
        if ending.is_some() {
            timestep.terminated.fill(true);
        }
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        let games = metrics.games.max(1.0);
        let traitors = self.config.num_traitors;
        let crew = self.config.num_agents - traitors;
        let per = |total: f64, agents: usize| total / games / agents.max(1) as f64;
        serde_json::json!({
            "reward": per(metrics.crew_reward + metrics.traitor_reward, self.config.num_agents),
            "crew_reward": per(metrics.crew_reward, crew),
            "traitor_reward": per(metrics.traitor_reward, traitors),
            "games": metrics.games,
            "wins_by_chores": metrics.wins_by_chores / games,
            "wins_by_vote": metrics.wins_by_vote / games,
            "traitor_wins": metrics.traitor_wins / games,
            "game_steps": metrics.game_steps / games,
            "chore_presses": metrics.presses / games,
            "kills": metrics.kills / games,
            "meetings": metrics.meetings / games,
            "kicked_crew": metrics.kicked_crew / games,
            "kicked_traitors": metrics.kicked_traitors / games,
        })
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        self.config.num_agents
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        GridRenderSettings {
            obs_vocab: self.obs_vocab.clone(),
            tile_width: self.geometry.inner.x as usize,
            tile_height: self.geometry.inner.y as usize,
            view_width: self.config.view_width as usize,
            view_height: self.config.view_height as usize + UI_HEIGHT,
            ui_height: UI_HEIGHT,
        }
    }

    /// The bird's-eye view shows what no one living sees whole: the map, who
    /// is hiding where, and the ghosts.
    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let Geometry { pad, inner, .. } = self.geometry;
        let dim = (inner.x as usize, inner.y as usize);
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != dim {
            *tilemap = Array2::zeros(dim);
        }
        let interior = self.state.map.slice(s![
            pad.x as usize..(pad.x + inner.x) as usize,
            pad.y as usize..(pad.y + inner.y) as usize,
        ]);
        tilemap.zip_mut_with(&interior, |dst, &tile| *dst = tile.into());

        grid_render_state.agent_positions.clear();
        for member in &self.state.members {
            let local = member.position - pad;
            grid_render_state.agent_positions.push(local);

            let under = self.state.map[member.position.idx()];
            match member.status {
                Status::Alive
                    if self.state.base_map[member.position.idx()] == MutinyObs::HidingSpot =>
                {
                    tilemap[local.idx()] = MutinyObs::HidingSpotOccupied.into();
                }
                Status::Dead if matches!(under, MutinyObs::TileEmpty | MutinyObs::TileWall) => {
                    tilemap[local.idx()] = MutinyObs::Ghost.into();
                }
                _ => {}
            }
        }
    }

    fn num_tasks(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests;
