import React from "react";
import "../ChatBox.css";
import { animations } from "react-animation";
import bot from "../../../../assets/bot.png";
import Avtar from "./AvtarChat";

const BotReply = (props) => {
  return (
    <div
      key={props.index}
      className="responseContainer"
      style={{ animation: animations.popIn }}
    >
      <Avtar icon={bot} />
      <div className="replyBot">
        <p>{props.text}</p>
      </div>
    </div>
  );
};

export default BotReply;
