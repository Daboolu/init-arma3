class CfgPatches
{
    class Checkerboard
    {
        units[] = {"Checkerboard"};
        weapons[] = {};
        requiredVersion = 0.1;
        requiredAddons[] = {};
    };
};

class CfgVehicles
{
    class Thing;

    class Checkerboard_8 : Thing
    {
        scope = 2;
        displayName = "Checkerboard";
        icon = "\Checkerboard\data\checkerboard_icon.paa";
        model = "\Checkerboard\data\checkerboard_8x8.p3d";
        destrType = "DestructNo";
        author = "Fangge Mao";
        mapSize = 16.0;
        descriptionShort = "For stereo camera calibration";
        cost = 10;
    };
};
