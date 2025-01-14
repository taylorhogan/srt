import leggo

l = leggo.Leggo()


@l.module(False)
def zwo6200mm (name, input)->{}:
    return True

@l.module(False)
def qhy6000m(name, input)->{}:
    return True

@l.module(False)
def zwoFocuser(name, input)->{}:
    return True

@l.module(False)
def qhyfilter7(name, input)->{}:
    return True

@l.module(False)
def geminiFocuser(name, input)->{}:
    return True

def geminiRotator(name, input)->{}:
    return True

@l.module(False)
def powerBox(name, input)->{}:
    return (True)

@l.module(False)
def gemini_controller(name, input)->{}:
    return (True)

@l.module(False)
def usb_hub(name, input)->{}:
    return (True)


@l.module(False)
def beelink(name, input)->{}:
    return (True)

@l.module(False)
def rpi4(name, input)->{}:
    return (True)

@l.module(False)
def oag(name, input)->{}:
    return True

@l.module(True)
def cdk17(name, input)->{}:
    powerBox("power box", input)
    qhy6000m("camera", input)
    qhyfilter7("filter", input)
    geminiFocuser("focuser", input)
    geminiRotator("rotator", input)
    oag("guider",input)

    return (True)

@l.module(True)
def rasa8(name, input)->{}:
    zwoFocuser("focuser", input)
    zwo6200mm("camera", input)
    return (True)


@l.module(True)
def l500(name, input)->{}:
    cdk17("cdk17", input)
    rasa8("rasa8", input)
    return True


@l.module(False)
def vision_safety(name, input)->{}:
    return True


@l.module(False)
def roof_motor(name, input) -> {}:
    return True


@l.module(False)
def kasa_wifi_switch(name, input) -> {}:
    return True

@l.module(False)
def limit_switch(name, input) -> {}:
    return True

@l.module(False)
def observatory_control(name, input)->{}:
    roof_motor("roof_motor", input)
    limit_switch("limit_switch_open", input)
    limit_switch("limit_switch_closed", input)
    kasa_wifi_switch("wifi_switch", input)
    return True



@l.module(True)
def iris(name, input)->{}:
    l500("mount", input)
    vision_safety("vision safety", input)
    observatory_control("observatory control", input)
    beelink("imaging computer", input)
    rpi4("master_control", input)

    return True


iris ("iris", {})
l.print_me()
l.render('iris')
