# Curriculum Pair Examples

## easy

1. similarity: `-0.0282`
   - A: A crash cymbal is struck.
     - dataset/wav: `fsd50k_dev` / `16257.wav`
     - labels: Crash_cymbal, Cymbal
   - B: A crowd screams and shouts.
     - dataset/wav: `fsd50k_dev` / `68058.wav`
     - labels: Crowd, Screaming, Shout
   - shared labels: none

2. similarity: `0.0373`
   - A: Fingers are snapping.
     - dataset/wav: `fsd50k_dev` / `257916.wav`
     - labels: Finger_snapping, Hands
   - B: Birds chirp, guns fire, someone shouts, and an explosion occurs.
     - dataset/wav: `fsd50k_eval` / `86346.wav`
     - labels: Bird, Bird_vocalization_and_bird_call_and_bird_song, Chirp_and_tweet, Explosion, Gunshot_and_gunfire, Shout
   - shared labels: none

3. similarity: `0.0238`
   - A: An explosion produces a booming fire sound.
     - dataset/wav: `fsd50k_dev` / `322509.wav`
     - labels: Boom, Explosion, Fire
   - B: A clock ticks to an alarm.
     - dataset/wav: `fsd50k_eval` / `189332.wav`
     - labels: Alarm, Clock, Mechanisms, Tick-tock
   - shared labels: none

## medium

1. similarity: `0.1003`
   - A: Gunshots and fireworks explode together.
     - dataset/wav: `fsd50k_dev` / `40971.wav`
     - labels: Explosion, Fireworks, Gunshot_and_gunfire
   - B: Traffic noise suggests the movement of vehicles on a road.
     - dataset/wav: `fsd50k_dev` / `106785.wav`
     - labels: Car, Traffic_noise_and_roadway_noise
   - shared labels: none

2. similarity: `0.1205`
   - A: Woodwind music flows harmoniously.
     - dataset/wav: `fsd50k_dev` / `119271.wav`
     - labels: Music, Musical_instrument, Wind_instrument_and_woodwind_instrument
   - B: Liquid gurgles, indicating flow or movement.
     - dataset/wav: `fsd50k_dev` / `317656.wav`
     - labels: Gurgling
   - shared labels: none

3. similarity: `0.1109`
   - A: Dishes, pots, and pans create clattering sounds at home.
     - dataset/wav: `fsd50k_dev` / `97559.wav`
     - labels: Dishes_and_pots_and_pans
   - B: Scissors are cutting.
     - dataset/wav: `fsd50k_dev` / `50599.wav`
     - labels: Scissors
   - shared labels: none

## hard

1. similarity: `0.1920`
   - A: Cutlery and dishes clink together softly.
     - dataset/wav: `fsd50k_eval` / `193060.wav`
     - labels: Chink_and_clink, Cutlery_and_silverware, Dishes_and_pots_and_pans, Glass
   - B: A melodious tune emanates from a string instrument.
     - dataset/wav: `fsd50k_dev` / `354274.wav`
     - labels: Bowed_string_instrument
   - shared labels: none

2. similarity: `0.2618`
   - A: A bell rings clearly.
     - dataset/wav: `fsd50k_dev` / `148849.wav`
     - labels: Bell
   - B: A ringtone combines elements of music and alarm.
     - dataset/wav: `fsd50k_dev` / `77928.wav`
     - labels: Alarm, Keyboard_(musical), Ringtone, Telephone
   - shared labels: none

3. similarity: `0.2285`
   - A: An engine produces mechanical noise.
     - dataset/wav: `fsd50k_dev` / `232935.wav`
     - labels: Engine
   - B: A door squeaks, screeches, and slams in a home.
     - dataset/wav: `fsd50k_eval` / `98152.wav`
     - labels: Door, Screech, Slam, Squeak
   - shared labels: none

## sim_0.30_0.50

1. similarity: `0.3083`
   - A: A human voice is heard.
     - dataset/wav: `fsd50k_eval` / `272357.wav`
     - labels: Human_voice
   - B: Whispered voices are softly heard.
     - dataset/wav: `fsd50k_dev` / `236392.wav`
     - labels: Whispering
   - shared labels: none

2. similarity: `0.3130`
   - A: The bass drum booms deeply.
     - dataset/wav: `fsd50k_dev` / `193561.wav`
     - labels: Bass_drum, Drum
   - B: A crash cymbal introduces dramatic percussive elements.
     - dataset/wav: `fsd50k_eval` / `19475.wav`
     - labels: Crash_cymbal, Cymbal
   - shared labels: none

3. similarity: `0.3443`
   - A: Melodies flow from a woodwind instrument.
     - dataset/wav: `fsd50k_dev` / `359970.wav`
     - labels: Music, Musical_instrument, Wind_instrument_and_woodwind_instrument
   - B: A trumpet plays boldly.
     - dataset/wav: `fsd50k_dev` / `357351.wav`
     - labels: Brass_instrument, Trumpet
   - shared labels: none

## sim_0.50_0.70

1. similarity: `0.5687`
   - A: A guitar produces melodious tunes.
     - dataset/wav: `fsd50k_dev` / `329507.wav`
     - labels: Guitar
   - B: A harp plays melodious music with gentle plucks.
     - dataset/wav: `fsd50k_eval` / `397937.wav`
     - labels: Harp
   - shared labels: none

2. similarity: `0.5257`
   - A: A man speaks as something cracks.
     - dataset/wav: `fsd50k_eval` / `145362.wav`
     - labels: Crack, Male_speech_and_man_speaking
   - B: Crushing noises occur repeatedly.
     - dataset/wav: `fsd50k_dev` / `336623.wav`
     - labels: Crushing
   - shared labels: none

3. similarity: `0.5485`
   - A: Coins clink against glass.
     - dataset/wav: `fsd50k_dev` / `341404.wav`
     - labels: Coin_(dropping), Glass
   - B: Keys jangle, creating a domestic sound.
     - dataset/wav: `fsd50k_eval` / `145383.wav`
     - labels: Keys_jangling
   - shared labels: none

## sim_gt_0.70

1. similarity: `0.7099`
   - A: Water flows from a tap.
     - dataset/wav: `fsd50k_eval` / `122818.wav`
     - labels: Water_tap_and_faucet
   - B: Liquid fills a container.
     - dataset/wav: `fsd50k_dev` / `206016.wav`
     - labels: Fill_(with_liquid)
   - shared labels: none

2. similarity: `0.7335`
   - A: An animal emits a noise.
     - dataset/wav: `fsd50k_dev` / `236957.wav`
     - labels: Animal
   - B: A bird makes a sound.
     - dataset/wav: `fsd50k_dev` / `189530.wav`
     - labels: Bird
   - shared labels: none

3. similarity: `0.7044`
   - A: An organ is being played.
     - dataset/wav: `fsd50k_dev` / `373684.wav`
     - labels: Keyboard_(musical), Organ
   - B: A wind instrument contributes to an orchestral piece.
     - dataset/wav: `fsd50k_dev` / `121477.wav`
     - labels: Music, Musical_instrument, Wind_instrument_and_woodwind_instrument
   - shared labels: none
