from Backend.cap_model import Model

if __name__ == '__main__':
    model = Model('dir')
    rep = model.get_report('history')
    print(rep)
