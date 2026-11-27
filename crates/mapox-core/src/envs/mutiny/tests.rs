use super::*;
use crate::policy::{Policy, RandomPolicy};
use crate::timestep::TimeStepBuffers;
use MutinyAction::*;
use MutinyObs::*;
use ndarray::Axis;
use std::collections::HashSet;

/// Eight aboard the frigate, two of them traitors, with an episode no test
/// outlives.
fn frigate() -> MutinyConfig {
    MutinyConfig {
        map: MutinyMap::Frigate,
        ..Default::default()
    }
}

fn start(config: MutinyConfig) -> (Mutiny, TimeStepBuffers) {
    let mut env = Mutiny::new(&config, 1 << 20);
    let mut buffers = TimeStepBuffers::new(&env);
    env.reset(7, &mut buffers.view_mut());
    (env, buffers)
}

/// The map cell of column `x` of row `row` of the frigate's drawing.
fn deck(x: i32, row: i32) -> Position {
    Geometry::new(&frigate()).ship_cell(&SHIPS[0], x, row)
}

fn with_role(env: &Mutiny, role: Role) -> Vec<usize> {
    (0..env.num_agents())
        .filter(|&id| env.state.members[id].role == role)
        .collect()
}

/// The traitors' ids, then the loyal crew's.
fn sides(env: &Mutiny) -> (Vec<usize>, Vec<usize>) {
    (with_role(env, Role::Traitor), with_role(env, Role::Crew))
}

/// Moves a living agent to an open cell by hand.
fn place(env: &mut Mutiny, id: usize, position: Position) {
    assert_eq!(env.state.map[position.idx()], TileEmpty, "{position:?}");
    env.erase_members();
    env.state.members[id].position = position;
    env.paint_members();
}

/// Steps the env with the listed agents' actions; everyone else waits.
fn step(env: &mut Mutiny, buffers: &mut TimeStepBuffers, moves: &[(usize, MutinyAction)]) {
    let mut actions = vec![VocabId::from(Noop); env.num_agents()];
    for &(id, action) in moves {
        actions[id] = action.into();
    }
    env.step(&actions, &mut buffers.view_mut());
}

/// Encodes the current state without stepping it.
fn observe(env: &Mutiny) -> TimeStepBuffers {
    let mut buffers = TimeStepBuffers::new(env);
    env.encode_observations(&mut buffers.view_mut());
    env.encode_action_mask(&mut buffers.view_mut());
    buffers
}

/// What `viewer`'s observation shows at the map cell `position`.
fn seen(env: &Mutiny, buffers: &TimeStepBuffers, viewer: usize, position: Position) -> MutinyObs {
    let center = Position::new(env.config.view_width / 2, env.config.view_height / 2);
    let cell = position - env.state.members[viewer].position + center;
    MutinyObs::from_id(buffers.obs[[viewer, cell.x as usize, cell.y as usize, 0]])
}

/// Row `row` of `id`'s UI band, `0` the lower one.
fn band(env: &Mutiny, buffers: &TimeStepBuffers, id: usize, row: usize) -> Vec<MutinyObs> {
    let y = env.config.view_height as usize + row;
    buffers
        .obs
        .slice(s![id, .., y, 0])
        .iter()
        .map(|&tile| MutinyObs::from_id(tile))
        .collect()
}

fn legal(buffers: &TimeStepBuffers, id: usize) -> Vec<MutinyAction> {
    MutinyAction::TABLE
        .iter()
        .copied()
        .filter(|&action| buffers.action_mask[[id, action as usize]])
        .collect()
}

/// The move that takes an agent at `from` into the cell beside it at `to`.
fn toward(from: Position, to: Position) -> MutinyAction {
    [MoveUp, MoveRight, MoveDown, MoveLeft]
        .into_iter()
        .find(|action| from + action.step().unwrap() == to)
        .expect("the cells are side by side")
}

/// An open cell beside the fitting at `position`.
fn beside(env: &Mutiny, position: Position) -> Position {
    STEPS
        .iter()
        .map(|step| position + step.to_pos())
        .find(|&cell| env.state.map[cell.idx()] == TileEmpty)
        .expect("a fitting has open deck beside it")
}

/// Stands `id` by its lit button and walks it into it.
fn press_lit(env: &mut Mutiny, buffers: &mut TimeStepBuffers, id: usize) -> Position {
    let button = env.state.members[id].lit.expect("a chore to do");
    let from = beside(env, button);
    place(env, id, from);
    step(env, buffers, &[(id, toward(from, button))]);
    button
}

/// Makes `id` able to attack now.
fn arm(env: &mut Mutiny, id: usize) {
    env.state.members[id].attack_cooldown = 0;
}

/// Where walking casts `ballot` in the vote room.
fn button_for(env: &Mutiny, ballot: Ballot) -> Position {
    match ballot {
        Ballot::Skip => env.geometry.skip_button(),
        Ballot::Kick(id) => vote_button_of(env, id),
    }
}

/// In a vote, stands `id` in front of the button for `ballot` and walks it
/// into it.
fn cast(env: &mut Mutiny, buffers: &mut TimeStepBuffers, id: usize, ballot: Ballot) {
    let button = button_for(env, ballot);
    // whoever stands in front of it is no obstacle in here
    env.erase_members();
    let front = beside(env, button);
    env.state.members[id].position = front;
    env.paint_members();
    step(env, buffers, &[(id, toward(front, button))]);
}

fn at(env: &Mutiny, id: usize) -> Position {
    env.state.members[id].position
}

fn vote_button_of(env: &Mutiny, id: usize) -> Position {
    env.geometry.vote_button(env.state.members[id].color)
}

/// Who `viewer` sees marked on the vote room's boards as having called the
/// vote.
fn marked(env: &Mutiny, buffers: &TimeStepBuffers, viewer: usize) -> Vec<usize> {
    (0..env.num_agents())
        .filter(|&id| {
            let mark = env.geometry.caller_mark(vote_button_of(env, id));
            match seen(env, buffers, viewer, mark) {
                UiCaller => true,
                UI => false,
                tile => panic!("{tile:?} where a caller is marked"),
            }
        })
        .collect()
}

#[test]
fn a_game_deals_everyone_a_color_a_role_and_chores() {
    let (env, buffers) = start(frigate());
    let members = &env.state.members;

    let (traitors, crew) = sides(&env);
    assert_eq!((traitors.len(), crew.len()), (2, 6));

    let colors: HashSet<u8> = members.iter().map(|member| member.color).collect();
    assert_eq!(colors.len(), 8, "no two wear the same color");
    let cells: HashSet<Position> = members.iter().map(|member| member.position).collect();
    assert_eq!(cells.len(), 8, "no two start on the same cell");

    for (id, member) in members.iter().enumerate() {
        assert_eq!(member.position, member.spawn);
        assert_eq!(
            env.state.map[member.position.idx()],
            MutinyObs::crew(member.color)
        );
        assert_eq!(env.state.by_color[usize::from(member.color)], Some(id));

        if member.role == Role::Traitor {
            assert!(member.chores.is_empty());
            assert_eq!(member.lit, None, "a traitor is shown no button to press");
            continue;
        }
        // three chores of four presses each, in three different rooms
        let rooms: HashSet<usize> = member.chores.iter().map(|chore| chore.room).collect();
        assert_eq!((member.chores.len(), rooms.len()), (3, 3));
        let lit = member.lit.expect("a button is lit");
        assert!(env.state.rooms[member.chores[0].room].contains(&lit));
    }
    // only the loyal crew's presses are owed
    assert_eq!(env.state.chores_left, 6 * 3 * 4);
    assert!(buffers.reward.iter().all(|&reward| reward == 0.0));
    assert!(buffers.terminated.iter().all(|&terminated| !terminated));
}

/// Roles and colors are dealt again each game, so neither can be learnt
/// from an agent's id.
#[test]
fn roles_and_colors_change_from_game_to_game() {
    let (mut env, mut buffers) = start(frigate());
    let mut traitors = HashSet::new();
    let mut colors = HashSet::new();
    for seed in 0..24 {
        env.reset(seed, &mut buffers.view_mut());
        traitors.extend(with_role(&env, Role::Traitor));
        colors.insert(env.state.members[0].color);
    }
    assert_eq!(traitors.len(), 8, "everyone gets a turn as a traitor");
    assert!(colors.len() > 4, "agent 0 wore only {colors:?}");
}

#[test]
fn the_band_shows_a_crew_member_its_role_and_the_counts() {
    let (env, buffers) = start(frigate());
    let (_, crew) = sides(&env);

    let mut top = vec![UI; 15];
    top[0] = UiRoleCrew;
    (top[9], top[11]) = (UiDeadCrew, Digit0);
    (top[12], top[14]) = (UiDeadTraitors, Digit0);
    assert_eq!(band(&env, &buffers, crew[0], 1), top);

    // 72 presses owed, the way to its own next one, and nothing counting
    // down outside a vote
    let member = &env.state.members[crew[0]];
    let mut bottom = vec![UI; 15];
    (bottom[0], bottom[2], bottom[3]) = (UiChores, Digit7, Digit2);
    bottom[5] = MutinyObs::arrow(member.lit.unwrap() - member.position);
    assert_eq!(band(&env, &buffers, crew[0], 0), bottom);
}

/// The arrow in a loyal crew member's band points at its lit button from
/// wherever it stands, to the nearest eighth of a turn. A traitor has none,
/// and no one has during a vote.
#[test]
fn the_band_points_a_crew_member_at_its_lit_button() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let me = crew[0];
    let from = deck(15, 7);
    place(&mut env, me, from);

    let pointer = |env: &Mutiny, id: usize| band(env, &observe(env), id, 0)[5];
    for ((x, y), arrow) in [
        ((0, 9), ArrowUp),
        ((9, 9), ArrowUpRight),
        ((9, 0), ArrowRight),
        ((9, -9), ArrowDownRight),
        ((0, -9), ArrowDown),
        ((-9, -9), ArrowDownLeft),
        ((-9, 0), ArrowLeft),
        ((-9, 9), ArrowUpLeft),
        // either side of the lines between up, up-right and right
        ((2, 5), ArrowUp),
        ((5, 12), ArrowUpRight),
        ((12, 5), ArrowUpRight),
        ((5, 2), ArrowRight),
        // rooms away, and far out of sight
        ((-14, 20), ArrowUpLeft),
        ((1, -17), ArrowDown),
    ] {
        env.state.members[me].lit = Some(from + Position::new(x, y));
        assert_eq!(pointer(&env, me), arrow, "({x}, {y})");
    }

    // it follows the chore from one button to the next
    env.light_next(me);
    let button = press_lit(&mut env, &mut buffers, me);
    let member = &env.state.members[me];
    let next = member.lit.unwrap();
    assert_ne!(next, button);
    assert_eq!(pointer(&env, me), MutinyObs::arrow(next - member.position));

    assert_eq!(pointer(&env, traitors[0]), UiAttack);
    env.start_vote();
    assert_eq!(pointer(&env, me), UI);

    // and with no chores left there is nowhere to point
    env.end_vote(&mut buffers.view_mut());
    env.state.members[me].lit = None;
    assert_eq!(pointer(&env, me), UI);
}

/// A traitor's band has what a crew member's has, and its fellow traitor
/// drawn as it looks on the map, plus its own attack cooldown.
#[test]
fn the_band_shows_a_traitor_who_its_fellows_are() {
    let (mut env, buffers) = start(frigate());
    let (traitors, _) = sides(&env);
    let (me, fellow) = (traitors[0], traitors[1]);
    let fellow_color = env.state.members[fellow].color;

    let top = band(&env, &buffers, me, 1);
    assert_eq!(top[0], UiRoleTraitor);
    assert_eq!(top[1], MutinyObs::crew(fellow_color));
    assert_eq!(top[2], UI);
    assert_eq!(
        &top[9..],
        &[UiDeadCrew, UI, Digit0, UiDeadTraitors, UI, Digit0]
    );

    let bottom = band(&env, &buffers, me, 0);
    assert_eq!(&bottom[5..8], &[UiAttack, Digit3, Digit0]);

    // a dead fellow is drawn as its corpse
    env.die(fellow);
    let buffers = observe(&env);
    assert_eq!(
        band(&env, &buffers, me, 1)[1],
        MutinyObs::corpse(fellow_color)
    );
    assert_eq!(
        &band(&env, &buffers, me, 1)[9..],
        &[UiDeadCrew, UI, Digit0, UiDeadTraitors, UI, Digit1]
    );
}

/// Walking into the lit button presses it: the crew owe one press fewer,
/// another button in the room lights, and everyone sees the press for a step.
#[test]
fn walking_into_the_lit_button_does_a_chore() {
    let (mut env, mut buffers) = start(frigate());
    let (_, crew) = sides(&env);
    let (worker, watcher) = (crew[0], crew[1]);

    let button = env.state.members[worker].lit.unwrap();
    let room = env.state.members[worker].chores[0].room;
    let from = beside(&env, button);
    place(&mut env, worker, from);
    let nearby = beside(&env, from);
    place(&mut env, watcher, nearby);
    // the watcher's own chore is somewhere else for this test
    env.state.members[watcher].lit = None;

    let before = observe(&env);
    assert_eq!(seen(&env, &before, worker, button), ChoreButtonLit);
    assert_eq!(seen(&env, &before, watcher, button), ChoreButton);

    step(&mut env, &mut buffers, &[(worker, toward(from, button))]);
    let member = &env.state.members[worker];
    assert_eq!(member.position, from, "a button is not walked onto");
    assert_eq!(env.state.chores_left, 71);
    assert_eq!(buffers.reward[worker], env.config.chore_reward);
    assert_eq!(buffers.reward[watcher], 0.0);
    assert_eq!(member.chores[0].presses_left, 3);

    let next = member.lit.unwrap();
    assert_ne!(next, button, "the same button never lights twice running");
    assert!(env.state.rooms[room].contains(&next));
    assert_eq!(seen(&env, &buffers, worker, button), ChoreButtonPressed);
    assert_eq!(seen(&env, &buffers, watcher, button), ChoreButtonPressed);
    assert_eq!(seen(&env, &buffers, worker, next), ChoreButtonLit);

    step(&mut env, &mut buffers, &[]);
    assert_eq!(seen(&env, &buffers, watcher, button), ChoreButton);
}

/// A chore's presses move round its room, and the chore after it is in
/// another room.
#[test]
fn a_finished_chore_gives_way_to_the_next() {
    let (mut env, mut buffers) = start(frigate());
    let worker = sides(&env).1[0];
    let rooms: Vec<usize> = env.state.members[worker]
        .chores
        .iter()
        .map(|chore| chore.room)
        .collect();

    for press in 0..12 {
        let button = press_lit(&mut env, &mut buffers, worker);
        assert!(env.state.rooms[rooms[press / 4]].contains(&button));
    }
    assert_eq!(env.state.chores_left, 60);
    assert!(env.state.members[worker].chores.is_empty());
    assert_eq!(env.state.members[worker].lit, None, "nothing left to do");
}

/// A button that is not lit for whoever walks into it does nothing.
#[test]
fn an_unlit_button_does_nothing() {
    let (mut env, mut buffers) = start(frigate());
    let worker = sides(&env).1[0];
    let lit = env.state.members[worker].lit.unwrap();
    let other = *env.state.rooms[env.state.members[worker].chores[0].room]
        .iter()
        .find(|&&button| button != lit)
        .unwrap();

    let from = beside(&env, other);
    place(&mut env, worker, from);
    step(&mut env, &mut buffers, &[(worker, toward(from, other))]);

    assert_eq!(env.state.chores_left, 72);
    assert_eq!(buffers.reward[worker], 0.0);
    assert_eq!(env.state.members[worker].lit, Some(lit));
    assert_eq!(seen(&env, &buffers, worker, other), ChoreButton);
}

/// A traitor has no chores, but any chore button it walks into shows as
/// pressed to everyone for a step, the same one twice running included, and
/// counts for nothing.
#[test]
fn a_traitors_presses_are_for_show() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (traitor, watcher) = (traitors[0], crew[0]);
    env.state.members[watcher].lit = None;

    let rooms = env.state.rooms[..env.state.num_rooms].to_vec();
    for button in rooms.into_iter().flatten() {
        let from = beside(&env, button);
        place(&mut env, traitor, from);
        let nearby = beside(&env, from);
        place(&mut env, watcher, nearby);

        for _ in 0..2 {
            step(&mut env, &mut buffers, &[(traitor, toward(from, button))]);
            assert_eq!(env.state.members[traitor].position, from);
            assert_eq!(seen(&env, &buffers, traitor, button), ChoreButtonPressed);
            assert_eq!(seen(&env, &buffers, watcher, button), ChoreButtonPressed);
            assert_eq!(buffers.reward[traitor], 0.0);
            assert_eq!(env.state.chores_left, 72);
            assert_eq!(env.state.members[traitor].lit, None);
        }

        step(&mut env, &mut buffers, &[]);
        assert_eq!(seen(&env, &buffers, watcher, button), ChoreButton);
    }
}

/// An attack takes no aim: it kills the loyal crew on every side of the
/// traitor at once, leaves a corpse of each, and spares the other traitor.
#[test]
fn an_attack_kills_the_crew_all_around() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (killer, fellow) = (traitors[0], traitors[1]);
    let (left, diagonal, far) = (crew[0], crew[1], crew[2]);

    place(&mut env, killer, deck(15, 16));
    place(&mut env, fellow, deck(16, 16));
    place(&mut env, left, deck(14, 16));
    place(&mut env, diagonal, deck(16, 17));
    place(&mut env, far, deck(13, 16));

    // nothing happens while the attack cools down
    assert!(!legal(&observe(&env), killer).contains(&Attack));
    step(&mut env, &mut buffers, &[(killer, Attack)]);
    assert_eq!(env.state.dead_crew, 0);

    arm(&mut env, killer);
    assert!(legal(&observe(&env), killer).contains(&Attack));
    step(&mut env, &mut buffers, &[(killer, Attack)]);

    for (victim, fell) in [(left, deck(14, 16)), (diagonal, deck(16, 17))] {
        let member = &env.state.members[victim];
        assert_eq!(member.status, Status::Dead);
        assert_eq!(
            seen(&env, &buffers, far, fell),
            MutinyObs::corpse(member.color)
        );
    }
    assert_eq!(env.state.members[fellow].status, Status::Alive);
    assert_eq!(env.state.members[far].status, Status::Alive);
    assert_eq!(env.state.dead_crew, 2);
    assert_eq!(buffers.reward[killer], 2.0 * env.config.kill_reward);
    assert_eq!(buffers.reward[fellow], 0.0);
    // each death costs every loyal crew member, the two killed included
    for &id in &crew {
        assert_eq!(buffers.reward[id], 2.0 * env.config.ally_killed_reward);
    }

    // the band counts the dead for everyone, and the cooldown for the killer
    assert_eq!(
        &band(&env, &buffers, far, 1)[9..12],
        &[UiDeadCrew, UI, Digit2]
    );
    assert_eq!(
        &band(&env, &buffers, killer, 0)[5..8],
        &[UiAttack, Digit2, Digit9]
    );
    assert!(!legal(&buffers, killer).contains(&Attack));
    assert!(!buffers.terminated[left], "the game goes on");
}

/// Attack is offered only with someone to kill, and never to the crew.
#[test]
fn an_attack_needs_a_victim_in_reach() {
    let (mut env, _) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (killer, fellow) = (traitors[0], traitors[1]);

    place(&mut env, killer, deck(15, 16));
    place(&mut env, fellow, deck(16, 16));
    place(&mut env, crew[0], deck(13, 16));
    arm(&mut env, killer);
    arm(&mut env, fellow);
    env.state.members[crew[0]].attack_cooldown = 0;

    let buffers = observe(&env);
    assert!(!legal(&buffers, killer).contains(&Attack));
    assert!(!legal(&buffers, crew[0]).contains(&Attack));

    place(&mut env, crew[0], deck(14, 17));
    let buffers = observe(&env);
    assert!(legal(&buffers, killer).contains(&Attack));
    assert!(!legal(&buffers, fellow).contains(&Attack), "too far away");
    assert!(!legal(&buffers, crew[1]).contains(&Attack));
}

/// A longer reach still needs a line of sight: the hull shelters whoever is
/// behind it.
#[test]
fn the_hull_stops_an_attack() {
    let (mut env, mut buffers) = start(MutinyConfig {
        attack_radius: 2,
        ..frigate()
    });
    let (traitors, crew) = sides(&env);
    let (sheltered, exposed) = (crew[0], crew[1]);

    // row 6 of the frigate is hull between the corridor and the top rooms
    place(&mut env, traitors[0], deck(2, 7));
    place(&mut env, sheltered, deck(2, 5));
    place(&mut env, exposed, deck(4, 7));
    arm(&mut env, traitors[0]);
    step(&mut env, &mut buffers, &[(traitors[0], Attack)]);

    assert_eq!(env.state.members[sheltered].status, Status::Alive);
    assert_eq!(env.state.members[exposed].status, Status::Dead);
}

/// The dead are out of the game: off the map with only a corpse left on it,
/// shown a blank observation, and able to do nothing.
#[test]
fn the_dead_are_off_the_map_and_shown_nothing() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (killer, first, second, witness) = (traitors[0], crew[0], crew[1], crew[2]);

    place(&mut env, killer, deck(15, 16));
    place(&mut env, first, deck(14, 16));
    place(&mut env, second, deck(16, 16));
    place(&mut env, witness, deck(12, 17));
    arm(&mut env, killer);
    let chores = env.state.chores_left;
    step(&mut env, &mut buffers, &[(killer, Attack)]);

    let blank = |buffers: &TimeStepBuffers, id: usize| {
        let obs = buffers.obs.slice(s![id, .., .., 0]);
        let (view, band) = obs.split_at(Axis(1), 15);
        view.iter().all(|&tile| tile == VocabId::from(Mask))
            && band.iter().all(|&tile| tile == VocabId::from(UI))
    };
    for dead in [first, second] {
        assert_eq!(at(&env, dead), OFF_MAP);
        assert!(blank(&buffers, dead));
    }
    assert!(!blank(&buffers, witness));
    assert!(!blank(&buffers, killer));

    // the living see the corpses, and nothing else of the dead
    for (dead, fell) in [(first, deck(14, 16)), (second, deck(16, 16))] {
        let corpse = MutinyObs::corpse(env.state.members[dead].color);
        assert_eq!(seen(&env, &buffers, witness, fell), corpse);
    }

    // the dead have no chores: their own passed to the living
    assert_eq!(env.state.members[first].lit, None);
    assert_eq!(env.state.chores_left, chores, "the presses are still owed");
    let owed: u32 = crew
        .iter()
        .flat_map(|&id| &env.state.members[id].chores)
        .map(|chore| chore.presses_left)
        .sum();
    assert_eq!(owed, chores);

    // a move is all the dead are offered, and it takes them nowhere
    assert_eq!(
        legal(&buffers, first),
        vec![MoveUp, MoveRight, MoveDown, MoveLeft, Noop]
    );
    let map = env.state.map.clone();
    for action in [MoveUp, MoveRight, MoveDown, MoveLeft, Attack] {
        step(&mut env, &mut buffers, &[(first, action)]);
        assert_eq!(at(&env, first), OFF_MAP);
        assert!(blank(&buffers, first));
    }
    assert_eq!(env.state.map, map);
    assert_eq!(env.state.phase, Phase::Play);
}

#[test]
fn only_traitors_can_walk_onto_hiding_tiles() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (hider, fellow, watcher, victim) = (traitors[0], traitors[1], crew[0], crew[1]);

    // the hiding spot in the frigate's top middle room
    let spot = deck(11, 3);
    let door = deck(12, 3);
    let below = deck(11, 4);
    place(&mut env, hider, door);
    place(&mut env, fellow, deck(14, 2));
    place(&mut env, watcher, deck(13, 4));
    place(&mut env, victim, below);

    step(&mut env, &mut buffers, &[(victim, MoveUp)]);
    assert_eq!(at(&env, victim), below);
    assert_eq!(env.state.map[spot.idx()], HidingSpot);

    step(&mut env, &mut buffers, &[(hider, MoveLeft)]);
    assert_eq!(at(&env, hider), spot);
    assert_eq!(seen(&env, &buffers, watcher, spot), HidingSpot);
    assert_eq!(seen(&env, &buffers, watcher, door), TileEmpty);
    assert_eq!(seen(&env, &buffers, fellow, spot), HidingSpot);
    // the tile covers the hider in its own view too
    assert_eq!(seen(&env, &buffers, hider, spot), HidingSpot);

    // An occupant can attack and remains concealed from other agents.
    arm(&mut env, hider);
    assert!(legal(&observe(&env), hider).contains(&Attack));
    step(&mut env, &mut buffers, &[(hider, Attack)]);
    assert!(!env.state.members[victim].alive());
    assert_eq!(seen(&env, &buffers, watcher, spot), HidingSpot);

    // the way out is blocked while someone stands in it
    place(&mut env, watcher, door);
    step(&mut env, &mut buffers, &[(hider, MoveRight)]);
    assert_eq!(at(&env, hider), spot);

    place(&mut env, watcher, deck(13, 4));
    step(&mut env, &mut buffers, &[(hider, MoveRight)]);
    assert_eq!(at(&env, hider), door);
    assert_eq!(env.state.map[spot.idx()], HidingSpot);
    assert_eq!(
        seen(&env, &buffers, watcher, door),
        MutinyObs::crew(env.state.members[hider].color)
    );
}

#[test]
fn adjacent_hiding_tiles_form_a_passage_with_normal_collision() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, _) = sides(&env);
    let (first, second) = (traitors[0], traitors[1]);
    let entry = deck(14, 3);
    let passage = [deck(13, 3), deck(12, 3), deck(11, 3)];
    let beside_passage = deck(12, 4);
    place(&mut env, first, entry);
    place(&mut env, second, beside_passage);
    for position in passage {
        env.state.base_map[position.idx()] = HidingSpot;
        env.state.map[position.idx()] = HidingSpot;
    }

    for position in passage {
        step(&mut env, &mut buffers, &[(first, MoveLeft)]);
        assert_eq!(at(&env, first), position);
        assert_eq!(seen(&env, &buffers, second, position), HidingSpot);
        if position == passage[1] {
            step(&mut env, &mut buffers, &[(second, MoveUp)]);
            assert_eq!(at(&env, second), beside_passage);
        }
    }

    step(&mut env, &mut buffers, &[(first, MoveUp), (second, MoveUp)]);
    assert_eq!(at(&env, first), deck(11, 2));
    assert_eq!(at(&env, second), passage[1]);
}

/// The meeting button is off for a while after the game starts. Once on,
/// walking into it takes everyone living to the vote room, the hidden
/// included, and clears the corpses away.
#[test]
fn the_meeting_button_calls_a_vote() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let caller = crew[0];
    let button = env.state.meeting_button;
    let from = deck(14, 11);
    assert_eq!(toward(from, button), MoveRight);

    place(&mut env, caller, from);
    place(&mut env, traitors[0], deck(15, 16));
    place(&mut env, crew[1], deck(14, 16));
    place(&mut env, traitors[1], deck(12, 3));
    arm(&mut env, traitors[0]);
    step(
        &mut env,
        &mut buffers,
        &[
            (traitors[0], Attack),
            (traitors[1], MoveLeft),
            (caller, MoveRight),
        ],
    );
    assert_eq!(env.state.phase, Phase::Play, "the button is still off");
    assert_eq!(at(&env, traitors[1]), deck(11, 3));
    assert_eq!(seen(&env, &buffers, caller, button), MeetingButtonOff);
    assert_eq!(env.state.members[caller].meetings_left, 1);
    let corpse = deck(14, 16);
    assert!(env.state.map[corpse.idx()].is_corpse());

    env.state.meeting_cooldown = 0;
    assert_eq!(seen(&env, &observe(&env), caller, button), MeetingButton);
    step(&mut env, &mut buffers, &[(caller, MoveRight)]);

    assert_eq!(env.state.phase, Phase::Vote { steps_left: 30 });
    assert_eq!(env.state.members[caller].meetings_left, 0);
    assert_eq!(env.state.map[corpse.idx()], TileEmpty);
    assert_eq!(env.state.map[deck(11, 3).idx()], HidingSpot);
    for (id, member) in env.state.members.iter().enumerate() {
        let place = if id == crew[1] {
            OFF_MAP
        } else {
            env.geometry.vote_floor(id)
        };
        assert_eq!(member.position, place);
    }
    assert_eq!(
        &band(&env, &buffers, caller, 0)[11..],
        &[UiVote, UI, Digit3, Digit0]
    );

    // everyone sees every living crew member's button with no votes on it,
    // and the dead one's corpse where its button would be
    for id in 0..env.num_agents() {
        let button = vote_button_of(&env, id);
        let (tile, count) = if id == crew[1] {
            (MutinyObs::corpse(env.state.members[id].color), UI)
        } else {
            (MutinyObs::vote_button(env.state.members[id].color), Digit0)
        };
        assert_eq!(seen(&env, &buffers, caller, button), tile);
        assert_eq!(
            seen(&env, &buffers, caller, env.geometry.tally(button)),
            count
        );
    }
    let skip = env.geometry.skip_button();
    assert_eq!(seen(&env, &buffers, caller, skip), SkipButton);
    assert_eq!(
        seen(&env, &buffers, caller, env.geometry.tally(skip)),
        Digit0
    );
    // and who it was that called the vote
    for viewer in (0..env.num_agents()).filter(|&id| id != crew[1]) {
        assert_eq!(marked(&env, &buffers, viewer), [caller]);
    }

    // no one can attack in the vote room
    arm(&mut env, traitors[0]);
    let during = observe(&env);
    for id in 0..env.num_agents() {
        assert_eq!(
            legal(&during, id),
            vec![MoveUp, MoveRight, MoveDown, MoveLeft, Noop]
        );
    }
}

/// A crew member has only so many meetings to call, and the button shows
/// off to one that has used them up.
#[test]
fn a_crew_member_runs_out_of_meetings() {
    let (mut env, mut buffers) = start(MutinyConfig {
        meeting_cooldown: 0,
        vote_steps: 1,
        ..frigate()
    });
    let (_, crew) = sides(&env);
    let (caller, other) = (crew[0], crew[1]);
    let button = env.state.meeting_button;

    place(&mut env, caller, deck(14, 11));
    step(&mut env, &mut buffers, &[(caller, MoveRight)]);
    step(&mut env, &mut buffers, &[]);
    assert_eq!(env.state.phase, Phase::Play);

    place(&mut env, caller, deck(14, 11));
    place(&mut env, other, deck(16, 11));
    let buffers = &mut observe(&env);
    assert_eq!(seen(&env, buffers, caller, button), MeetingButtonOff);
    assert_eq!(seen(&env, buffers, other, button), MeetingButton);

    step(&mut env, buffers, &[(caller, MoveRight)]);
    assert_eq!(env.state.phase, Phase::Play);
    step(&mut env, buffers, &[(other, MoveLeft)]);
    assert_eq!(env.state.phase, Phase::Vote { steps_left: 1 });
}

/// Finding a body calls a vote whatever the meeting button's state.
#[test]
fn walking_into_a_corpse_calls_a_vote() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let (killer, victim, finder) = (traitors[0], crew[0], crew[1]);

    place(&mut env, killer, deck(15, 16));
    place(&mut env, victim, deck(14, 16));
    place(&mut env, finder, deck(12, 16));
    arm(&mut env, killer);
    step(&mut env, &mut buffers, &[(killer, Attack)]);

    step(&mut env, &mut buffers, &[(finder, MoveRight)]);
    assert_eq!(env.state.phase, Phase::Play);

    step(&mut env, &mut buffers, &[(finder, MoveRight)]);
    assert_eq!(env.state.phase, Phase::Vote { steps_left: 30 });
    assert_eq!(env.state.members[finder].meetings_left, 1);
    assert!(env.state.corpses.is_empty());
    assert_eq!(marked(&env, &buffers, killer), [finder]);
}

/// Whoever calls a vote is marked on the board for as long as it lasts, a
/// traitor reporting its own kill like anyone else, and everyone who called
/// it on the same step with it.
#[test]
fn the_board_marks_whoever_called_the_vote() {
    let (mut env, mut buffers) = start(MutinyConfig {
        vote_steps: 2,
        ..frigate()
    });
    let (traitors, crew) = sides(&env);
    let (killer, finder) = (traitors[0], crew[2]);

    // a vote no one called has no one marked
    env.erase_members();
    env.start_vote();
    assert!(marked(&env, &observe(&env), finder).is_empty());
    env.end_vote(&mut buffers.view_mut());

    place(&mut env, killer, deck(15, 16));
    place(&mut env, crew[0], deck(14, 16));
    arm(&mut env, killer);
    step(&mut env, &mut buffers, &[(killer, Attack)]);
    step(&mut env, &mut buffers, &[(killer, MoveLeft)]);
    assert_eq!(env.state.phase, Phase::Vote { steps_left: 2 });
    assert_eq!(marked(&env, &buffers, finder), [killer]);

    // the mark stays through the vote and is gone with it
    step(&mut env, &mut buffers, &[]);
    assert_eq!(marked(&env, &buffers, finder), [killer]);
    step(&mut env, &mut buffers, &[]);
    assert_eq!(env.state.phase, Phase::Play);
    assert!(env.state.callers.is_empty());
    let board = env.geometry.caller_mark(vote_button_of(&env, killer));
    assert_eq!(env.state.map[board.idx()], UI);

    // two find the next body at once
    place(&mut env, killer, deck(15, 16));
    place(&mut env, crew[1], deck(14, 16));
    place(&mut env, finder, deck(13, 16));
    arm(&mut env, killer);
    step(&mut env, &mut buffers, &[(killer, Attack)]);
    step(
        &mut env,
        &mut buffers,
        &[(killer, MoveLeft), (finder, MoveRight)],
    );
    let mut both = [killer, finder];
    both.sort();
    assert_eq!(marked(&env, &buffers, crew[3]), both);
}

/// In the vote room the living walk through each other, and a vote shows in
/// the count by its button as soon as it is cast.
#[test]
fn votes_are_cast_by_walking_into_a_button() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let suspect = traitors[0];
    env.erase_members();
    env.die(crew[5]);
    env.start_vote();

    // two agents on one cell, each at the centre of its own view
    let shared = env.geometry.vote_floor(9);
    for id in [crew[0], crew[1]] {
        env.erase_members();
        env.state.members[id].position = shared;
        env.paint_members();
    }
    let together = observe(&env);
    for id in [crew[0], crew[1]] {
        assert_eq!(
            seen(&env, &together, id, shared),
            MutinyObs::crew(env.state.members[id].color)
        );
    }

    let button = vote_button_of(&env, suspect);
    let tally = env.geometry.tally(button);
    cast(&mut env, &mut buffers, crew[0], Ballot::Kick(suspect));
    assert_eq!(
        env.state.members[crew[0]].ballot,
        Some(Ballot::Kick(suspect))
    );
    assert_eq!(seen(&env, &buffers, crew[3], tally), Digit1);
    cast(&mut env, &mut buffers, crew[1], Ballot::Kick(suspect));
    assert_eq!(seen(&env, &buffers, crew[3], tally), Digit2);

    // a second vote, a vote for oneself and a vote for the dead do not count
    cast(&mut env, &mut buffers, crew[0], Ballot::Skip);
    assert_eq!(
        env.state.members[crew[0]].ballot,
        Some(Ballot::Kick(suspect))
    );
    cast(&mut env, &mut buffers, suspect, Ballot::Kick(suspect));
    assert_eq!(env.state.members[suspect].ballot, None);
    cast(&mut env, &mut buffers, crew[2], Ballot::Kick(crew[5]));
    assert_eq!(env.state.members[crew[2]].ballot, None);
    assert_eq!(seen(&env, &buffers, crew[3], tally), Digit2);

    let skip = env.geometry.skip_button();
    cast(&mut env, &mut buffers, crew[2], Ballot::Skip);
    assert_eq!(env.state.members[crew[2]].ballot, Some(Ballot::Skip));
    assert_eq!(
        seen(&env, &buffers, crew[3], env.geometry.tally(skip)),
        Digit1
    );
    assert_eq!(env.state.phase, Phase::Vote { steps_left: 24 });
}

/// Whoever has the most votes is kicked: it dies and leaves no corpse, the
/// count of the dead says which side it was on, and everyone else goes back
/// to where they started with the cooldowns reset.
#[test]
fn the_most_voted_is_kicked() {
    let (mut env, mut buffers) = start(MutinyConfig {
        vote_steps: 4,
        ..frigate()
    });
    let (traitors, crew) = sides(&env);
    let suspect = traitors[0];
    arm(&mut env, traitors[1]);
    env.erase_members();
    env.start_vote();

    cast(&mut env, &mut buffers, crew[0], Ballot::Kick(suspect));
    cast(&mut env, &mut buffers, crew[1], Ballot::Kick(suspect));
    cast(&mut env, &mut buffers, suspect, Ballot::Kick(crew[0]));
    assert_eq!(env.state.phase, Phase::Vote { steps_left: 1 });
    cast(&mut env, &mut buffers, crew[2], Ballot::Skip);

    assert_eq!(env.state.phase, Phase::Play);
    assert_eq!(env.state.members[suspect].status, Status::Dead);
    assert_eq!((env.state.dead_crew, env.state.dead_traitors), (0, 1));
    assert!(env.state.corpses.is_empty());
    assert_eq!(env.state.meeting_cooldown, 20);
    assert_eq!(env.state.members[traitors[1]].attack_cooldown, 30);
    for member in &env.state.members {
        let at_spawn = env.state.map[member.spawn.idx()];
        if member.alive() {
            assert_eq!(member.position, member.spawn);
            assert_eq!(at_spawn, MutinyObs::crew(member.color));
        } else {
            assert_eq!(member.position, OFF_MAP);
            assert_eq!(at_spawn, TileEmpty);
        }
    }
    assert_eq!(
        &band(&env, &buffers, crew[0], 1)[9..],
        &[UiDeadCrew, UI, Digit0, UiDeadTraitors, UI, Digit1]
    );
    assert!(!buffers.terminated[suspect], "one traitor is left");
}

/// A kick pays everyone on the side the kicked agent was not on, and docks
/// everyone on its own, the kicked and the dead among them, on the step the
/// vote ends.
#[test]
fn a_kick_pays_the_other_side_and_docks_its_own() {
    for kick_traitor in [true, false] {
        let (mut env, mut buffers) = start(MutinyConfig {
            vote_steps: 4,
            traitor_kicked_reward: 0.5,
            crew_kicked_reward: 0.125,
            ally_kicked_reward: -0.75,
            ..frigate()
        });
        let (traitors, crew) = sides(&env);
        let suspect = if kick_traitor { traitors[0] } else { crew[5] };
        env.erase_members();
        env.die(crew[4]);
        env.start_vote();

        for &voter in &crew[..3] {
            cast(&mut env, &mut buffers, voter, Ballot::Kick(suspect));
            assert!(buffers.reward.iter().all(|&reward| reward == 0.0));
        }
        cast(&mut env, &mut buffers, crew[3], Ballot::Skip);
        assert_eq!(env.state.phase, Phase::Play);
        assert_eq!(env.state.members[suspect].status, Status::Dead);

        let (crew_reward, traitor_reward) = if kick_traitor {
            (0.5, -0.75)
        } else {
            (-0.75, 0.125)
        };
        for id in 0..env.num_agents() {
            let expected = if crew.contains(&id) {
                crew_reward
            } else {
                traitor_reward
            };
            assert_eq!(buffers.reward[id], expected, "agent {id}");
        }
        let metrics = env.consume_metrics();
        assert_eq!(metrics["crew_reward"], crew_reward);
        assert_eq!(metrics["traitor_reward"], traitor_reward);
    }
}

/// A tie kicks no one, and neither does a lead no bigger than the skips.
#[test]
fn a_tie_or_a_skip_kicks_no_one() {
    for skips in [false, true] {
        let (mut env, mut buffers) = start(MutinyConfig {
            vote_steps: 4,
            ..frigate()
        });
        let (traitors, crew) = sides(&env);
        env.erase_members();
        env.start_vote();

        cast(&mut env, &mut buffers, crew[0], Ballot::Kick(traitors[0]));
        cast(&mut env, &mut buffers, crew[1], Ballot::Kick(traitors[0]));
        let rival = if skips {
            Ballot::Skip
        } else {
            Ballot::Kick(crew[0])
        };
        cast(&mut env, &mut buffers, crew[2], rival);
        cast(&mut env, &mut buffers, crew[3], rival);

        assert_eq!(env.state.phase, Phase::Play);
        assert!(env.state.members.iter().all(Member::alive));
    }
}

/// With every living vote in, there is nothing to wait for.
#[test]
fn a_vote_ends_once_everyone_living_has_voted() {
    let (mut env, mut buffers) = start(frigate());
    env.erase_members();
    env.die(0);
    env.start_vote();

    for id in 1..env.num_agents() {
        assert!(matches!(env.state.phase, Phase::Vote { .. }));
        cast(&mut env, &mut buffers, id, Ballot::Skip);
    }
    assert_eq!(env.state.phase, Phase::Play);
}

/// The last press of the last chore wins the game for the crew: everyone on
/// each side is paid, dead or alive, and a new game is dealt at once.
#[test]
fn the_crew_win_by_finishing_their_chores() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    let roles: Vec<Role> = env.state.members.iter().map(|member| member.role).collect();

    env.erase_members();
    env.die(crew[1]);
    env.paint_members();
    // one press of one chore is all the crew still owe
    for &id in &crew {
        env.state.members[id].chores.clear();
        env.state.members[id].lit = None;
    }
    let worker = crew[0];
    env.state.members[worker].chores.push_back(Chore {
        room: 0,
        presses_left: 1,
    });
    env.light_next(worker);
    env.state.chores_left = 1;
    press_lit(&mut env, &mut buffers, worker);

    for id in 0..env.num_agents() {
        let side = match roles[id] {
            Role::Crew => env.config.win_reward,
            Role::Traitor => env.config.loss_reward,
        };
        let chore = if id == worker {
            env.config.chore_reward
        } else {
            0.0
        };
        assert_eq!(buffers.reward[id], side + chore, "agent {id}");
        assert!(buffers.terminated[id]);
    }
    assert_eq!(buffers.reward[crew[1]], 1.0, "the dead win too");
    assert_eq!(buffers.reward[traitors[0]], -1.0);

    // the observation is the next game's first
    assert_eq!(env.state.game_time, 0);
    assert_eq!(env.state.chores_left, 72);
    assert!(env.state.members.iter().all(Member::alive));
    assert_eq!(buffers.time[0], 1, "the episode carries on");

    let metrics = env.consume_metrics();
    assert_eq!(metrics["games"], 1.0);
    assert_eq!(metrics["wins_by_chores"], 1.0);
    assert_eq!(metrics["traitor_wins"], 0.0);
}

/// Kicking the last traitor wins it too.
#[test]
fn the_crew_win_by_kicking_every_traitor() {
    let (mut env, mut buffers) = start(MutinyConfig {
        num_traitors: 1,
        ..frigate()
    });
    let (traitors, crew) = sides(&env);
    env.erase_members();
    env.start_vote();
    cast(&mut env, &mut buffers, crew[0], Ballot::Kick(traitors[0]));
    for &id in &crew[1..] {
        cast(&mut env, &mut buffers, id, Ballot::Skip);
    }
    assert!(buffers.terminated.iter().all(|&terminated| !terminated));
    cast(&mut env, &mut buffers, traitors[0], Ballot::Skip);

    // six skips outvote the one, so the traitor stays
    assert_eq!(env.state.phase, Phase::Play);
    assert!(buffers.terminated.iter().all(|&terminated| !terminated));

    env.erase_members();
    env.start_vote();
    for &id in &crew {
        if matches!(env.state.phase, Phase::Vote { .. }) {
            cast(&mut env, &mut buffers, id, Ballot::Kick(traitors[0]));
        }
    }
    cast(&mut env, &mut buffers, traitors[0], Ballot::Skip);
    assert!(buffers.terminated.iter().all(|&terminated| terminated));
    // the kick pays the crew on top of the win, and costs the traitor on
    // top of the loss
    assert_eq!(
        buffers.reward[crew[0]],
        1.0 + env.config.traitor_kicked_reward
    );
    assert_eq!(
        buffers.reward[traitors[0]],
        -1.0 + env.config.ally_kicked_reward
    );
    assert_eq!(env.consume_metrics()["wins_by_vote"], 1.0);
}

/// The traitors win once they are as many as the loyal crew left alive.
#[test]
fn the_traitors_win_by_matching_the_crew() {
    let (mut env, mut buffers) = start(MutinyConfig {
        num_agents: 4,
        num_traitors: 1,
        ..frigate()
    });
    let (traitors, crew) = sides(&env);
    let killer = traitors[0];

    place(&mut env, killer, deck(15, 16));
    place(&mut env, crew[0], deck(14, 16));
    place(&mut env, crew[1], deck(16, 16));
    arm(&mut env, killer);
    step(&mut env, &mut buffers, &[(killer, Attack)]);

    assert!(buffers.terminated.iter().all(|&terminated| terminated));
    assert_eq!(buffers.reward[killer], 1.0 + 2.0 * env.config.kill_reward);
    for &id in &crew {
        assert_eq!(
            buffers.reward[id],
            -1.0 + 2.0 * env.config.ally_killed_reward
        );
    }
    let metrics = env.consume_metrics();
    assert_eq!(metrics["traitor_wins"], 1.0);
    assert_eq!(metrics["kills"], 2.0);
}

/// A game has no clock of its own: however long everyone waits, it goes on
/// and no one is paid for it.
#[test]
fn waiting_wins_the_game_for_no_one() {
    let (mut env, mut buffers) = start(frigate());

    for _ in 0..2000 {
        step(&mut env, &mut buffers, &[]);
        assert!(buffers.terminated.iter().all(|&terminated| !terminated));
        assert!(buffers.reward.iter().all(|&reward| reward == 0.0));
    }
    // nothing counts down in the band outside a vote
    assert_eq!(&band(&env, &buffers, 0, 0)[11..], &[UI; 4]);
    assert_eq!(env.consume_metrics()["games"], 0.0);
}

/// The episode ends at `length` whatever the game is doing, and pays no one
/// for a game cut short.
#[test]
fn the_episode_ends_at_its_length() {
    let mut env = Mutiny::new(&frigate(), 3);
    let mut buffers = TimeStepBuffers::new(&env);
    env.reset(0, &mut buffers.view_mut());

    for time in 1..=3 {
        step(&mut env, &mut buffers, &[]);
        assert!(buffers.terminated.iter().all(|&done| done == (time == 3)));
        assert!(buffers.reward.iter().all(|&reward| reward == 0.0));
    }
}

/// A named ship is the only one dealt, and sizes the map on its own.
#[test]
fn a_named_ship_is_the_only_one_dealt() {
    for (map, ship) in [
        (MutinyMap::Frigate, &SHIPS[0]),
        (MutinyMap::Hauler, &SHIPS[1]),
        (MutinyMap::Station, &SHIPS[2]),
    ] {
        let (env, _) = start(MutinyConfig {
            map,
            ..Default::default()
        });
        let settings = env.get_render_settings();
        assert_eq!(settings.tile_width as i32, ship.width());
        assert_eq!(settings.tile_height as i32, ship.height() + VOTE_HEIGHT);

        let decks = ship
            .rows
            .iter()
            .flat_map(|row| row.bytes())
            .filter(|tile| matches!(tile, b'.' | b'S'))
            .count();
        let floor = env
            .state
            .base_map
            .iter()
            .filter(|&&tile| tile == TileEmpty)
            .count();
        assert_eq!(floor, decks + 24, "the ship's deck and the vote room's");
    }

    // left to chance, every ship turns up
    let (mut env, mut buffers) = start(MutinyConfig::default());
    let mut rooms = HashSet::new();
    for seed in 0..16 {
        env.reset(seed, &mut buffers.view_mut());
        rooms.insert(env.state.rooms.concat());
    }
    assert_eq!(rooms.len(), 3);
}

/// The vote room's constants describe the drawing they sit beside.
#[test]
fn the_vote_room_is_laid_out_as_drawn() {
    let (env, _) = start(frigate());
    let geometry = env.geometry;
    let drawn = |position: Position| {
        let local = position - geometry.vote_origin;
        VOTE_ROOM[(VOTE_HEIGHT - 1 - local.y) as usize].as_bytes()[local.x as usize]
    };

    for color in 0..MAX_CREW as u8 {
        let button = geometry.vote_button(color);
        assert_eq!(drawn(button), b'0' + color);
        assert!(VOTE_BOARDS.contains(&(button.y - geometry.vote_origin.y)));
        assert_eq!(drawn(geometry.tally(button)), b't');
        assert_eq!(drawn(geometry.caller_mark(button)), b'c');
    }
    assert_eq!(drawn(geometry.skip_button()), b's');
    assert_eq!(drawn(geometry.tally(geometry.skip_button())), b't');

    let floor: HashSet<Position> = (0..24).map(|index| geometry.vote_floor(index)).collect();
    assert_eq!(floor.len(), 24);
    assert!(floor.iter().all(|&cell| drawn(cell) == b'.'));
    let dots = VOTE_ROOM.concat().bytes().filter(|&b| b == b'.').count();
    assert_eq!(dots, 24);
}

/// Wherever an agent stands in the vote room it sees both boards whole,
/// whichever colors are out of the game: a count no one can read from the
/// far corner would be no way to follow a vote.
#[test]
fn the_whole_vote_room_is_in_sight_from_anywhere_in_it() {
    let (mut env, _) = start(MutinyConfig {
        num_agents: 3,
        num_traitors: 1,
        ..frigate()
    });
    env.erase_members();
    env.start_vote();
    let origin = env.geometry.vote_origin;

    for index in 0..24 {
        env.erase_members();
        env.state.members[0].position = env.geometry.vote_floor(index);
        env.paint_members();
        let buffers = observe(&env);
        for x in 0..VOTE_WIDTH {
            for y in 0..VOTE_HEIGHT {
                let cell = origin + Position::new(x, y);
                assert_ne!(seen(&env, &buffers, 0, cell), Mask, "{x},{y} from {index}");
            }
        }
    }
}

/// The bird's-eye view shows a hidden traitor like anyone else, and has the
/// dead off the map.
#[test]
fn the_render_state_shows_the_hidden_and_not_the_dead() {
    let (mut env, mut buffers) = start(frigate());
    let (traitors, crew) = sides(&env);
    place(&mut env, traitors[0], deck(12, 3));
    step(&mut env, &mut buffers, &[(traitors[0], MoveLeft)]);
    env.erase_members();
    env.die(crew[0]);
    env.paint_members();

    let mut render = GridRenderState::default();
    env.render_state_into(&mut render);
    let at = |id: usize| MutinyObs::from_id(render.tilemap[render.agent_positions[id].idx()]);
    for id in [traitors[0], crew[1]] {
        assert_eq!(at(id), MutinyObs::crew(env.state.members[id].color));
    }
    assert_eq!(render.agent_positions[crew[0]], OFF_MAP);
    assert_eq!(render.agent_positions.len(), 8);
}

/// Whole games of legal random play on every ship, checking after each step
/// that the map, the members and the chore count still agree, and that the
/// dead are off the map and shown nothing.
#[test]
fn random_play_keeps_the_books_straight() {
    for map in [MutinyMap::Frigate, MutinyMap::Hauler, MutinyMap::Station] {
        let config = MutinyConfig {
            map,
            num_agents: 10,
            num_traitors: 2,
            attack_cooldown: 6,
            meeting_cooldown: 4,
            meetings_per_agent: 3,
            vote_steps: 12,
            ..Default::default()
        };
        let mut env = Mutiny::new(&config, 1 << 20);
        let mut buffers = TimeStepBuffers::new(&env);
        let mut actions = vec![0; env.num_agents()];
        let mut policy = RandomPolicy::new();
        policy.reset(env.num_agents(), 3).unwrap();
        env.reset(3, &mut buffers.view_mut());

        let mut votes = 0;
        for _ in 0..6000 {
            policy.act(&buffers.view(), &mut actions).unwrap();
            env.step(&actions, &mut buffers.view_mut());
            let state = &env.state;
            votes += usize::from(matches!(state.phase, Phase::Vote { .. }));

            let mut open = HashSet::new();
            for (id, member) in state.members.iter().enumerate() {
                let obs = buffers.obs.slice(s![id, .., .., 0]);
                assert!(usize::from(*obs.iter().max().unwrap()) < MutinyObs::TABLE.len());
                let (view, band) = obs.split_at(Axis(1), 15);
                match member.status {
                    Status::Alive => {
                        open.insert(member.position);
                        assert!(state.map[member.position.idx()].is_crew());
                        let hiding = state.base_map[member.position.idx()] == HidingSpot;
                        if hiding {
                            assert_eq!(member.role, Role::Traitor);
                        }
                        // a hiding tile covers whoever is on it from itself too
                        let own = VocabId::from(MutinyObs::crew(member.color));
                        assert_eq!(view.iter().any(|&tile| tile == own), !hiding);
                    }
                    Status::Dead => {
                        assert_eq!(member.position, OFF_MAP);
                        assert!(view.iter().all(|&tile| tile == VocabId::from(Mask)));
                        assert!(band.iter().all(|&tile| tile == VocabId::from(UI)));
                    }
                }
            }

            let drawn = state.map.iter().filter(|tile| tile.is_crew()).count();
            assert_eq!(drawn, open.len());
            if state.phase == Phase::Play {
                let living = state.members.iter().filter(|m| m.status == Status::Alive);
                assert_eq!(open.len(), living.count(), "one to a cell");
            }

            let owed: u32 = state
                .members
                .iter()
                .filter(|member| member.role == Role::Crew && member.alive())
                .flat_map(|member| &member.chores)
                .map(|chore| chore.presses_left)
                .sum();
            assert_eq!(owed, state.chores_left);
        }

        let metrics = env.consume_metrics();
        assert!(metrics["games"].as_f64().unwrap() >= 10.0, "{metrics}");
        assert!(metrics["kills"].as_f64().unwrap() > 0.0, "{metrics}");
        assert!(metrics["meetings"].as_f64().unwrap() > 0.0, "{metrics}");
        assert!(votes > 0);
    }
}
