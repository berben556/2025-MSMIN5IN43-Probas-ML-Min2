package com.game.gametheory.model;

import java.util.HashSet;
import java.util.Set;

public class Detective extends Creature {

    private int round = 0;
    private boolean someoneCheated = false;

    // mémoire des comportements adverses
    private Set<Creature> cheaters = new HashSet<>();

    public Detective(Position p) {
        super(p);
    }

    @Override
    public Species getSpecies() {
        return Species.DETECTIVE;
    }

    /**
     * Détermine si le Detective se comporte comme Hawk
     */
    public boolean behavesAsHawkAgainst(Creature other) {

        // Phase de test (4 premiers tours)
        if (round == 0) return false; // Dove
        if (round == 1) return true;  // Hawk
        if (round == 2) return false; // Dove
        if (round == 3) return false; // Dove

        // Après analyse
        if (!someoneCheated) {
            return true; // Exploitation : Hawk toujours
        }

        // Copycat
        return cheaters.contains(other);
    }

    /**
     * Mémorise une triche adverse
     */
    public void observe(Creature other, boolean otherBehavedAsHawk) {
        if (otherBehavedAsHawk) {
            someoneCheated = true;
            cheaters.add(other);
        }
    }

    public void nextRound() {
        round++;
    }
}
