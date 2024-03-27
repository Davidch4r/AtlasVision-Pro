from geoguessr import Game

game = Game()
game.play_rounds(1000, show=False, learn=True, show_after=-1, learning_rate=0.0005, save_after=10)