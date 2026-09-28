"""Sizing-aware RLCard NLHE env: raise-TO an explicit chip target.

Stock RLCard only raises BY pot/half-pot. This subclass layer adds a raise-TO
action via ``step_raise_to``; stock enum actions still flow through ``step``.
"""

from __future__ import annotations

import numpy as np
from rlcard.envs.nolimitholdem import DEFAULT_GAME_CONFIG, NolimitholdemEnv
from rlcard.envs.registration import DEFAULT_CONFIG as ENV_DEFAULT_CONFIG
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem import Game as NolimitholdemGame
from rlcard.games.nolimitholdem.round import Action, NolimitholdemRound

# Sized raises record as an aggressive enum so verb/pot_type/aggressor inference
# (aggression-vs-passivity only) treats them as raises.
_RAISE_TO_RECORD_ACTION = Action.RAISE_POT


class SizedNolimitholdemRound(NolimitholdemRound):
    """Adds raise-TO semantics to the stock round."""

    def min_raise_to(self) -> int:
        """Smallest legal raise-to (chips): max commitment + one big blind."""
        return max(self.raised) + self.init_raise_amount

    def max_raise_to(self, players: list | None = None) -> int:
        """Largest legal raise-to (chips) for the actor: all-in (current + stack)."""
        plist = players if players is not None else getattr(self, "_players", None)
        if plist is None:
            raise ValueError("max_raise_to: no players bound; pass players=")
        actor = plist[self.game_pointer]
        return self.raised[self.game_pointer] + actor.remained_chips

    def proceed_round_raise_to(self, players: list, target_raised: int) -> int:
        """Raise the actor's street commitment TO ``target_raised`` chips.

        Mirrors ``proceed_round``'s raise-branch bookkeeping but raise-TO rather
        than raise-BY. ``player.bet`` caps the spend at the stack (all-in clamp).
        """
        # Keep a handle so max_raise_to() works without re-threading players.
        self._players = players
        player = players[self.game_pointer]

        max_before = max(self.raised)
        additional = target_raised - self.raised[self.game_pointer]
        if additional < 0:
            additional = 0
        # player.bet caps the spend at remaining chips (all-in clamp).
        spent_before = player.in_chips
        player.bet(chips=additional)
        actually_spent = player.in_chips - spent_before
        self.raised[self.game_pointer] += actually_spent

        # A strict raise resets the no-raise counter; a non-raising target (call/limp
        # to the current level) accumulates toward is_over like a CHECK_CALL.
        if self.raised[self.game_pointer] > max_before:
            self.not_raise_num = 1
        else:
            self.not_raise_num += 1

        if player.remained_chips < 0:
            raise Exception("Player in negative stake")
        if player.remained_chips == 0 and player.status != PlayerStatus.FOLDED:
            player.status = PlayerStatus.ALLIN

        self.game_pointer = (self.game_pointer + 1) % self.num_players

        if player.status == PlayerStatus.ALLIN:
            self.not_playing_num += 1
            self.not_raise_num -= 1

        while players[self.game_pointer].status == PlayerStatus.FOLDED:
            self.game_pointer = (self.game_pointer + 1) % self.num_players

        return self.game_pointer


class SizedNolimitholdemGame(NolimitholdemGame):
    """NLHE game whose round supports raise-TO, plus a sized step entry point."""

    def init_game(self) -> tuple[dict, int]:
        state, player_id = super().init_game()
        # Replace the stock round with the sized subclass, preserving its state.
        sized = SizedNolimitholdemRound(
            self.num_players, self.big_blind, dealer=self.dealer, np_random=self.np_random
        )
        sized.__dict__.update(self.round.__dict__)
        sized._players = self.players
        self.round = sized
        return state, player_id

    def step_raise_to(self, target_raised: int) -> tuple[dict, int]:
        """Apply a raise-TO ``target_raised`` then run RLCard's street progression.

        The post-proceed block is duplicated verbatim from the parent ``step``
        (the lib source cannot be edited) to guarantee dealing/round-advance parity.
        """
        self.game_pointer = self.round.proceed_round_raise_to(self.players, target_raised)

        players_in_bypass = [
            1 if player.status in (PlayerStatus.FOLDED, PlayerStatus.ALLIN) else 0 for player in self.players
        ]
        if self.num_players - sum(players_in_bypass) == 1:
            last_player = players_in_bypass.index(0)
            if self.round.raised[last_player] >= max(self.round.raised):
                players_in_bypass[last_player] = 1

        if self.round.is_over():
            self.game_pointer = (self.dealer_id + 1) % self.num_players
            if sum(players_in_bypass) < self.num_players:
                while players_in_bypass[self.game_pointer]:
                    self.game_pointer = (self.game_pointer + 1) % self.num_players

            from rlcard.games.nolimitholdem.game import Stage

            if self.round_counter == 0:
                self.stage = Stage.FLOP
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            if self.round_counter == 1:
                self.stage = Stage.TURN
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            if self.round_counter == 2:
                self.stage = Stage.RIVER
                self.public_cards.append(self.dealer.deal_card())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)
        return state, self.game_pointer


class SizedNolimitholdemEnv(NolimitholdemEnv):
    """NLHE env exposing ``step_raise_to`` alongside the stock enum-action step."""

    def __init__(self, config: dict) -> None:
        self.name = "no-limit-holdem"
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = SizedNolimitholdemGame()
        # Skip NolimitholdemEnv.__init__ (it rebinds self.game = stock Game); call
        # the grandparent Env init so the sized game survives, then redo env setup.
        from rlcard.envs import Env

        Env.__init__(self, config)
        self.actions = Action
        self.state_shape = [[54] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]
        import json
        import os

        import rlcard as _rlcard

        with open(os.path.join(_rlcard.__path__[0], "games/limitholdem/card2index.json")) as file:
            self.card2index = json.load(file)

    def step_raise_to(self, target_raised: int) -> tuple[dict, int]:
        """Record a sized raise (as an aggressive enum) and step the sized game."""
        actor = self.get_player_id()
        all_in = target_raised >= self.game.round.max_raise_to(self.game.players)
        record_action = Action.ALL_IN if all_in else _RAISE_TO_RECORD_ACTION
        self.timestep += 1
        self.action_recorder.append((actor, record_action))
        next_state, player_id = self.game.step_raise_to(target_raised)
        return self._extract_state(next_state), player_id


def make_sized_env(config: dict | None = None) -> SizedNolimitholdemEnv:
    """Construct a SizedNolimitholdemEnv, merging RLCard env + NLHE game defaults
    (since we instantiate the env class directly, bypassing the registry)."""
    merged = dict(ENV_DEFAULT_CONFIG)
    merged.update(DEFAULT_GAME_CONFIG)
    if config:
        merged.update(config)
    return SizedNolimitholdemEnv(merged)
