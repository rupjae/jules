import Head from 'next/head';
import { Chat } from '../components/Chat';

export default function Home() {
  return (
    <>
      <Head>
        <title>Jules Chatbot</title>
      </Head>
      <Chat />
    </>
  );
}

