package com.game.gametheory.engine;

import com.game.gametheory.model.*;
import java.util.*;

public class GameEngine {
  public int day = 0;
  public Board board;

  public GameEngine(int h, int d, int g, int det) {
    board = new Board(280);
    for (int i = 0; i < h; i++) board.creatures.add(new Hawk(board.randomPosition()));
    for (int i = 0; i < d; i++) board.creatures.add(new Dove(board.randomPosition()));
    for (int i = 0; i < g; i++) board.creatures.add(new Grudge(board.randomPosition()));
    for (int i = 0; i < det; i++) board.creatures.add(new Detective(board.randomPosition()));
  }

  public GameSnapshot nextDay() {
    day++;
    List<Creature> next = new ArrayList<>();

    for (Creature c : board.creatures) {
      c.resetFood();

      // Simuler nourriture aléatoire
      double r = Math.random();
      if (r < 0.33) c.addFood(2);
      else if (r < 0.66) c.addFood(1);
      else c.addFood(0.5);

      // Gérer rancunier qui se fait arnaquer par Hawk
      if (c instanceof Grudge) {
        for (Creature other : board.creatures) {
          if (other.getSpecies() == Species.HAWK && c.getFood() < 1) {
            ((Grudge) c).rememberHawk(other);
          }
        }
      }

      // Gestion Detective
      if (c instanceof Detective) {
        Detective det = (Detective) c;

        for (Creature other : board.creatures) {
          if (other == c) continue;

          boolean otherIsHawk = other.getSpecies() == Species.HAWK
                  || (other instanceof Grudge && ((Grudge) other).behavesAsHawkAgainst(c));

          det.observe(other, otherIsHawk);
        }

        det.nextRound();
      }

      // Vérifier survie
      if (c.survives()) {
        next.add(c);

        // Vérifier reproduction
        if (c.reproduces()) {
          if (c.getSpecies() == Species.HAWK) next.add(new Hawk(board.randomPosition()));
          else if (c.getSpecies() == Species.DOVE) next.add(new Dove(board.randomPosition()));
          else if (c.getSpecies() == Species.GRUDGE) next.add(new Grudge(board.randomPosition()));
          else if (c.getSpecies() == Species.DETECTIVE) next.add(new Detective(board.randomPosition()));
        }
      }
    }

    board.creatures = next;
    return GameSnapshot.from(board.creatures, day);
  }
}